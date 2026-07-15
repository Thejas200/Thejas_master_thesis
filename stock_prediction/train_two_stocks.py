

import os
import math
import time
import json
import random
import shutil
import argparse
import subprocess
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reduce TF noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ------------------------------- Reproducibility -------------------------------
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

# Make GPU memory growth (prevents huge upfront alloc)
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# ------------------------------- IO / Paths -----------------------------------
ROOT_OUT_DIR = "ga_markov_outputs_lstm"
OUT_DIR = ROOT_OUT_DIR
FIG_DIR = os.path.join(OUT_DIR, "figs")

# ------------------------------- Data -----------------------------------------
PARQUET_PATH = "sp500_close_stooq.parquet"
USE_STREAMLIT_SELECTION = True
SELECTION_PATH = "selected_pair.json"
TARGET_STOCK = "AAPL"
PARTNER_STOCK = "MSFT"
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.json")
FEATURE_VOL_WINDOW = 20
FEATURE_MA_WINDOW = 20


def configure_output_dirs(out_dir: str, clear_existing: bool = True):
    # These globals are reused by many helper functions, so update them in one place
    global OUT_DIR, FIG_DIR, SUMMARY_PATH
    OUT_DIR = out_dir
    FIG_DIR = os.path.join(OUT_DIR, "figs")
    SUMMARY_PATH = os.path.join(OUT_DIR, "summary.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    # Remove tabular artifacts from an older run so generations from separate
    # searches cannot be mistaken for one continuous run.
    if clear_existing:
        for name in os.listdir(OUT_DIR):
            is_generation_csv = name.startswith("gen_") and name.endswith("_population.csv")
            if is_generation_csv or name in {"generation_summary.csv", "summary.json"}:
                try:
                    os.remove(os.path.join(OUT_DIR, name))
                except OSError:
                    pass
    if clear_existing and os.path.isdir(FIG_DIR):
        try:
            shutil.rmtree(FIG_DIR)
        except Exception:
            pass
    os.makedirs(FIG_DIR, exist_ok=True)


def clean(t: str) -> str:
    # Normalize ticker text so "brk.b", " BRK.B ", and similar variants map consistently.
    return str(t).strip().upper().replace(".", "-")


def load_selected_pair(selection_path: str = SELECTION_PATH) -> Tuple[str, str]:
    # Read the pair saved by Streamlit so the trainer can run without manual re-entry.
    if not os.path.exists(selection_path):
        raise FileNotFoundError(
            f"{selection_path} not found. Run streamlit app and save a target/partner pair first."
        )
    with open(selection_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    target = clean(payload["target"])
    partner = clean(payload["partner"])
    return target, partner


def resolve_stock_pair(
    use_streamlit_selection: bool = USE_STREAMLIT_SELECTION,
    target_stock: str = TARGET_STOCK,
    partner_stock: str = PARTNER_STOCK,
    selection_path: str = SELECTION_PATH,
) -> Tuple[str, str]:
    # The training script can either consume the pair saved by Streamlit
    # or fall back to tickers passed directly on the command line.
    if use_streamlit_selection:
        target, partner = load_selected_pair(selection_path=selection_path)
        print(f"Using Streamlit-selected pair: {target}, {partner}")
        return target, partner
    target, partner = clean(target_stock), clean(partner_stock)
    print(f"Using manual pair from notebook config: {target}, {partner}")
    return target, partner


def load_close_price_frame(parquet_path: str, stocks: List[str]) -> pd.DataFrame:
    # Load the parquet once, standardize ticker names, and keep only the requested columns.
    df = pd.read_parquet(parquet_path)
    df.columns = [clean(c) for c in df.columns]
    df = df.dropna(axis=0, how="any")
    cleaned = [clean(s) for s in stocks]
    missing = [s for s in cleaned if s not in df.columns]
    if missing:
        raise ValueError(
            f"Missing selected stock(s) in dataset: {missing}"
        )
    unique_cleaned = list(dict.fromkeys(cleaned))
    return df[unique_cleaned].copy()


def load_stock_close_prices(parquet_path: str, stock: str) -> pd.DataFrame:
    # Keep the single-stock helper for plots and reporting.
    stock = clean(stock)
    return load_close_price_frame(parquet_path, [stock])[[stock]].copy()


def resolve_pair_context_for_stock(
    stock: str,
    use_streamlit_selection: bool,
    target_stock: str,
    partner_stock: str,
    selection_path: str,
) -> Tuple[str, str]:
    # For each training run, identify the "other" stock so pair based features stay consistent.
    target, partner = resolve_stock_pair(
        use_streamlit_selection=use_streamlit_selection,
        target_stock=target_stock,
        partner_stock=partner_stock,
        selection_path=selection_path,
    )
    stock = clean(stock)
    if stock == target:
        return target, partner
    if stock == partner:
        return partner, target
    raise ValueError(f"Stock {stock} is not part of the selected pair ({target}, {partner}).")


def build_feature_frame(parquet_path: str, stock: str, partner_stock: str) -> pd.DataFrame:
    # Derive a compact, higher signal feature set from the two chosen close-price series.
    stock = clean(stock)
    partner_stock = clean(partner_stock)
    pair_df = load_close_price_frame(parquet_path, [stock, partner_stock])
    stock_close = pair_df[stock].astype(np.float32)
    partner_close = pair_df[partner_stock].astype(np.float32)
    stock_log_return = np.log(stock_close / stock_close.shift(1))
    partner_log_return = np.log(partner_close / partner_close.shift(1))
    stock_roll_vol_20 = stock_log_return.rolling(FEATURE_VOL_WINDOW).std()
    stock_ma_20 = stock_close.rolling(FEATURE_MA_WINDOW).mean()
    stock_ma_gap_20 = (stock_close / stock_ma_20) - 1.0

    feature_df = pd.DataFrame(
        {
            "target_close": stock_close,
            "partner_close": partner_close,
            "target_log_return": stock_log_return,
            "partner_log_return": partner_log_return,
            "target_roll_vol_20": stock_roll_vol_20,
            "target_ma_gap_20": stock_ma_gap_20,
        },
        index=pair_df.index,
    ).dropna()
    if len(feature_df) < 200:
        raise ValueError(
            f"Too few rows remain after feature engineering for {stock}. "
            f"Need a longer history or fewer rolling features."
        )
    return feature_df


def make_sequences(
    arr2d: np.ndarray,
    target_arr: Optional[np.ndarray] = None,
    time_step: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled features -> return X: (M, time_step, F), y: (M, output_dim)."""
    if target_arr is None:
        target_arr = arr2d
    X, y = [], []
    for i in range(len(arr2d) - time_step):
        # Each training example contains `time_step` past values and predicts the next one.
        X.append(arr2d[i:i+time_step, :])
        y.append(target_arr[i+time_step])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y


def compute_sequence_split_counts(n_sequences: int, train_frac: float, val_frac: float) -> Tuple[int, int, int]:
    # Keep the split logic in one place so reporting matches the actual tensors.
    n_train = int(n_sequences * train_frac)
    n_val = int(n_sequences * val_frac)
    n_test = n_sequences - n_train - n_val
    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    if n_test < 1:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
    return n_train, n_val, n_test


def _format_index_value(idx_value) -> str:
    if isinstance(idx_value, pd.Timestamp):
        return idx_value.strftime("%Y-%m-%d")
    return str(idx_value)


def build_dataset_split_summary(
    df: pd.DataFrame,
    stock: str,
    time_step: int,
    train_frac: float,
    val_frac: float,
) -> Dict:
    # Summarize the chronological split in sequence space so it reflects model training.
    n_rows = len(df)
    n_sequences = n_rows - time_step
    if n_sequences < 3:
        raise ValueError(
            f"Not enough data for time_step={time_step}. Need > {time_step + 2} rows, got {n_rows}."
        )

    n_train, n_val, n_test = compute_sequence_split_counts(n_sequences, train_frac, val_frac)
    target_index = df.index[time_step:]

    def make_section(label: str, start: int, count: int) -> Dict:
        idx = target_index[start:start + count]
        return {
            "label": label,
            "sequence_count": int(count),
            "sequence_pct": float((count / n_sequences) * 100.0),
            "target_start": _format_index_value(idx[0]),
            "target_end": _format_index_value(idx[-1]),
        }

    return {
        "stock": stock,
        "time_step": int(time_step),
        "total_rows": int(n_rows),
        "total_sequences": int(n_sequences),
        "raw_start": _format_index_value(df.index[0]),
        "raw_end": _format_index_value(df.index[-1]),
        "train": make_section("train", 0, n_train),
        "val": make_section("val", n_train, n_val),
        "test": make_section("test", n_train + n_val, n_test),
    }


def save_dataset_split_plot(df: pd.DataFrame, stock: str, split_summary: Dict, out_dir: str) -> str:
    # Plot the full close price history with train/val/test regions overlaid.
    plot_path = os.path.join(out_dir, f"dataset_split_{stock}.png")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df[stock].to_numpy(), color="black", linewidth=1.1, label=f"{stock} close")

    for key, color in [("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")]:
        info = split_summary[key]
        start = pd.to_datetime(info["target_start"])
        end = pd.to_datetime(info["target_end"])
        ax.axvspan(start, end, color=color, alpha=0.18, label=f"{info['label']} ({info['sequence_count']} seq)")

    ax.set_title(f"Dataset Split Overview: {stock}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def save_dataset_split_detail_plot(df: pd.DataFrame, stock: str, split_summary: Dict, out_dir: str) -> str:
    # Provide a more detailed split view with separate train/val/test panels.
    plot_path = os.path.join(out_dir, f"dataset_split_detail_{stock}.png")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharey=True)
    sections = [
        ("train", "tab:blue"),
        ("val", "tab:orange"),
        ("test", "tab:green"),
    ]

    for ax, (key, color) in zip(axes, sections):
        info = split_summary[key]
        start = pd.to_datetime(info["target_start"])
        end = pd.to_datetime(info["target_end"])
        mask = (df.index >= start) & (df.index <= end)
        section_df = df.loc[mask]

        ax.plot(section_df.index, section_df[stock].to_numpy(), color=color, linewidth=1.4)
        ax.fill_between(
            section_df.index,
            section_df[stock].to_numpy(),
            np.min(section_df[stock].to_numpy()),
            color=color,
            alpha=0.10,
        )
        ax.set_title(
            f"{info['label'].upper()} | {info['sequence_count']} sequences "
            f"({info['sequence_pct']:.1f}%) | {info['target_start']} to {info['target_end']}"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Detailed Dataset Split Representation: {stock}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def save_learning_curve_plot(history, stock: str, stage_label: str, out_dir: str) -> Optional[str]:
    # Save train/validation curves for both loss and MSE when available.
    if history is None or not getattr(history, "history", None):
        return None

    hist = history.history
    epochs = np.arange(1, len(hist.get("loss", [])) + 1)
    if len(epochs) == 0:
        return None

    safe_stage = stage_label.lower().replace(" ", "_")
    plot_path = os.path.join(out_dir, f"{safe_stage}_learning_curve_{stock}.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, hist["loss"], label="train_loss", linewidth=1.5)
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], label="val_loss", linewidth=1.5)
    axes[0].set_title(f"{stage_label} Loss: {stock}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if "mse" in hist:
        axes[1].plot(epochs, hist["mse"], label="train_mse", linewidth=1.5)
    if "val_mse" in hist:
        axes[1].plot(epochs, hist["val_mse"], label="val_mse", linewidth=1.5)
    axes[1].set_title(f"{stage_label} MSE: {stock}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)
    if "mse" in hist or "val_mse" in hist:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def save_prediction_phase_plot(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    stock: str,
    phase_label: str,
    out_dir: str,
) -> str:
    # Save prediction vs true, zoomed view, residuals, and rolling RMSE for one split.
    safe_phase = phase_label.lower().replace(" ", "_")
    plot_path = os.path.join(out_dir, f"prediction_{safe_phase}_{stock}.png")
    true_values = np.asarray(true_values, dtype=float).reshape(-1)
    pred_values = np.asarray(pred_values, dtype=float).reshape(-1)
    errors = pred_values - true_values
    rolling_window = int(min(20, len(errors)))
    if rolling_window >= 2:
        rolling_rmse = np.sqrt(
            pd.Series(errors).pow(2).rolling(rolling_window, min_periods=rolling_window).mean()
        ).to_numpy()
    else:
        rolling_rmse = np.array([], dtype=float)

    zoom_n = int(min(80, len(true_values)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes.ravel()

    ax[0].plot(true_values, label=f"True {stock}", linewidth=1.5)
    ax[0].plot(pred_values, label=f"Pred {stock}", linewidth=1.2)
    ax[0].set_title(f"Full {phase_label} Prediction: {stock}")
    ax[0].set_xlabel(f"{phase_label} timestep")
    ax[0].set_ylabel("Close Price")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(true_values[-zoom_n:], label=f"True {stock}", linewidth=1.5)
    ax[1].plot(pred_values[-zoom_n:], label=f"Pred {stock}", linewidth=1.2)
    ax[1].set_title(f"Zoom (last {zoom_n} steps)")
    ax[1].set_xlabel("Relative timestep")
    ax[1].set_ylabel("Close Price")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    ax[2].plot(errors, color="tab:red", linewidth=1.2, label="Residual (pred-true)")
    ax[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax[2].set_title("Residuals")
    ax[2].set_xlabel(f"{phase_label} timestep")
    ax[2].set_ylabel("Error")
    ax[2].grid(True, alpha=0.3)
    ax[2].legend()

    if rolling_window >= 2:
        ax[3].plot(rolling_rmse, color="tab:green", linewidth=1.3, label=f"Rolling RMSE ({rolling_window})")
        ax[3].set_title("Rolling RMSE")
        ax[3].set_xlabel(f"{phase_label} timestep")
        ax[3].set_ylabel("RMSE")
        ax[3].grid(True, alpha=0.3)
        ax[3].legend()
    else:
        ax[3].text(0.5, 0.5, "Not enough points for rolling RMSE", ha="center", va="center")
        ax[3].set_axis_off()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path

def load_data_for_stock(
    stock: str,
    partner_stock: str,
    time_step=120,
    train_frac=0.8,
    val_frac=0.1,
    parquet_path=PARQUET_PATH,
):
    """
    Returns:
      X_tr, y_tr, X_val, y_val, X_te, y_te (scaled), target_scaler (for inverse-transform),
      y_te_inv (unscaled), feature_names, aligned_target_df
    """
    stock = clean(stock)
    partner_stock = clean(partner_stock)
    feature_df = build_feature_frame(parquet_path=parquet_path, stock=stock, partner_stock=partner_stock)
    raw_features = feature_df.values.astype(np.float32)
    raw_target = feature_df[["target_close"]].values.astype(np.float32)
    aligned_target_df = feature_df[["target_close"]].rename(columns={"target_close": stock})

    # Split first so the scaler only sees the training era and does not leak future prices.
    n_seq = len(raw_features) - time_step
    if n_seq < 3:
        raise ValueError(
            f"Not enough data for time_step={time_step}. Need > {time_step + 2} rows, got {len(raw_features)}."
        )
    n_train, n_val, n_test = compute_sequence_split_counts(n_seq, train_frac, val_frac)

    train_fit_end = n_train + time_step
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit on the training era only, then reuse the same scalers everywhere else.
    feature_scaler.fit(raw_features[:train_fit_end])
    target_scaler.fit(raw_target[:train_fit_end])
    scaled_features = feature_scaler.transform(raw_features)
    scaled_target = target_scaler.transform(raw_target)

    X_all, y_all = make_sequences(scaled_features, target_arr=scaled_target, time_step=time_step)
    n = len(X_all)
    if n < 1000:
        print(f"Warning: only {n} sequences formed. Consider reducing time_step or extending date range.")

    # Keep the split chronological because time series models must not shuffle future into past.
    n_train, n_val, n_test = compute_sequence_split_counts(n, train_frac, val_frac)

    X_tr = X_all[:n_train]
    y_tr = y_all[:n_train]
    X_val = X_all[n_train:n_train + n_val]
    y_val = y_all[n_train:n_train + n_val]
    X_te = X_all[n_train + n_val:]
    y_te = y_all[n_train + n_val:]

    # For reporting RMSE in price units (inverse transform)
    y_te_inv = target_scaler.inverse_transform(y_te)

    return (
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_te,
        y_te,
        target_scaler,
        y_te_inv,
        feature_df.columns.tolist(),
        aligned_target_df,
    )

# ------------------------------- Models ---------------------------------------
def build_baseline_lstm(sequence_length=120, num_features=2, output_dim=2, units=100, dropout=0.2, bidir=True):
    # This is the fixed reference model used before any GA optimization happens.
    model = keras.Sequential()
    model.add(keras.Input(shape=(sequence_length, num_features)))
    if bidir:
        model.add(layers.Bidirectional(layers.LSTM(units, return_sequences=True)))
        model.add(layers.Dropout(dropout))
        model.add(layers.Bidirectional(layers.LSTM(max(units // 2, 16))))
    else:
        model.add(layers.LSTM(units, return_sequences=True))
        model.add(layers.Dropout(dropout))
        model.add(layers.LSTM(max(units // 2, 16)))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_dim))
    # Adam + MSE is a standard setup for next step regression.
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mse"])
    return model

def build_model_from_params(p: Dict, sequence_length=120, num_features=2, output_dim=2):
    # Build an LSTM architecture from one GA candidate's parameter dictionary.
    # p keys: units, layers, dropout, bidir, batch_size
    model = keras.Sequential()
    model.add(keras.Input(shape=(sequence_length, num_features)))

    u = int(p["units"])
    num_layers = int(p["layers"])
    bidir = bool(p.get("bidir", True))
    dp = float(p["dropout"])

    # The first LSTM layer uses the widest representation; deeper layers taper down.
    if bidir:
        model.add(layers.Bidirectional(layers.LSTM(u, return_sequences=(num_layers > 1))))
    else:
        model.add(layers.LSTM(u, return_sequences=(num_layers > 1)))
    model.add(layers.Dropout(dp))

    # Additional layers shrink gradually to keep the search space expressive but stable.
    for i in range(1, num_layers):
        u = max(u // 2, 16)  # gentle taper
        is_last = (i == num_layers - 1)
        if bidir:
            model.add(layers.Bidirectional(layers.LSTM(u, return_sequences=not is_last)))
        else:
            model.add(layers.LSTM(u, return_sequences=not is_last))
        model.add(layers.Dropout(dp))

    model.add(layers.Dense(output_dim))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mse"])
    return model

def train_and_eval(model, X_tr, y_tr, X_val, y_val, epochs=20, batch_size=64, verbose=0):
    # Every GA candidate is judged by its best validation RMSE after early stopping.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        min_delta=1e-4,
    )
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[early_stop],
    )
    # Use best validation RMSE as objective (lower is better).
    val_mse = float(np.min(hist.history["val_mse"]))
    val_rmse = math.sqrt(val_mse)
    return val_rmse

# ------------------------------- GA Setup -------------------------------------
@dataclass
class Individual:
    # One GA individual = one candidate set of model hyperparameters.
    params: Dict
    fitness: Optional[float] = None  # We'll store NEGATIVE RMSE so "bigger is better"
    id: int = field(default=-1)

def sample_params() -> Dict:
    # Randomly create a new starting point inside the GA search space.
    return {
        "units": random.choice([64, 96, 128, 160, 192]),
        "layers": random.choice([1, 2, 3]),
        "dropout": round(float(np.clip(np.random.normal(0.25, 0.1), 0.05, 0.5)), 3),
        "batch_size": random.choice([32, 64, 96, 128]),
        "bidir": random.choice([0, 1]) == 1,
    }

def mutate_params(p: Dict) -> Dict:
    # Mutation changes some fields at random so the GA can keep exploring new ideas.
    q = dict(p)
    if random.random() < 0.4:
        q["units"] = random.choice([64, 96, 128, 160, 192])
    if random.random() < 0.4:
        q["layers"] = random.choice([1, 2, 3])
    if random.random() < 0.5:
        q["dropout"] = round(float(np.clip(q["dropout"] + np.random.normal(0, 0.05), 0.05, 0.55)), 3)
    if random.random() < 0.3:
        q["batch_size"] = random.choice([32, 64, 96, 128])
    if random.random() < 0.3:
        q["bidir"] = not q["bidir"]
    return q

def crossover_params(p1: Dict, p2: Dict) -> Dict:
    # Crossover mixes two parents by choosing each field from either parent.
    child = {}
    for k in p1.keys():
        child[k] = p1[k] if random.random() < 0.5 else p2[k]
    return child

def rank_bins(ranks: List[int], num_bins: int = 10) -> np.ndarray:
    """Map rank (0 best … n-1 worst) to bin index 0..num_bins-1."""
    n = len(ranks)
    bins = np.zeros(n, dtype=int)
    for i, r in enumerate(ranks):
        bins[i] = min((r * num_bins) // n, num_bins - 1)
    return bins

# --------------------- Markov utilities (robust & interpretable) ---------------------
def _smooth_and_normalize_rows(P: np.ndarray, counts: np.ndarray, alpha: float) -> np.ndarray:
    """
    Dirichlet/Laplace smoothing per row. For empty rows, fallback to uniform.
    Returns a valid row-stochastic matrix.
    """
    k = P.shape[1]
    Q = P.astype(float).copy()
    for i in range(Q.shape[0]):
        if counts[i] > 0:
            Q[i, :] = (Q[i, :] + alpha) / (counts[i] + alpha * k)
        else:
            Q[i, :] = 1.0 / k
    return Q

def _lazy_matrix(P: np.ndarray, lazy: bool) -> np.ndarray:
    # A lazy Markov chain blends the transition matrix with the identity matrix.
    # This can smooth the dynamics, but our code keeps it disabled by default.
    if not lazy:
        return P
    I = np.eye(P.shape[0], dtype=float)
    return 0.5 * (I + P)

def compute_markov_and_lambda2(
    parent_bins: List[int],
    child_bins: List[int],
    num_bins: int = 10,
    alpha: float = 1e-2,
    use_lazy: bool = False,  # default False (no lazy)
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Build smoothed (and optional lazy) transition matrix P, return:
      - P (row-stochastic),
      - lambda2 (SLEM),
      - relaxation_time,
      - counts per parent bin.
    """
    # P_raw counts how often a parent rank bin produces a child rank bin in one generation.
    P_raw = np.zeros((num_bins, num_bins), dtype=float)
    counts = np.zeros(num_bins, dtype=float)

    for pb, cb in zip(parent_bins, child_bins):
        # Count how often parent bin `pb` leads to child bin `cb`.
        P_raw[pb, cb] += 1.0
        counts[pb] += 1.0

    # Smooth + normalize (uniform fallback for empty rows)
    P = _smooth_and_normalize_rows(P_raw, counts, alpha=alpha)

    # Optional lazy chain to stabilize spectrum & avoid periodicity
    P = _lazy_matrix(P, lazy=use_lazy)

    # Eigenvalues: SLEM (second largest eigenvalue modulus)
    w = np.linalg.eigvals(P)
    w = np.sort(np.abs(w))[::-1]
    lambda2 = float(w[1]) if len(w) > 1 else 0.0

    # Relaxation time from the absolute spectral gap
    relaxation_time = float(1.0 / max(1e-9, (1.0 - min(lambda2, 0.999999))))

    return P, lambda2, relaxation_time, counts

def print_matrix(P):
    """Pretty print 10×10 transition matrix with parent/child labels."""
    with np.printoptions(precision=3, suppress=True):
        print(P)

def save_transition_heatmap(P: np.ndarray, gen: int, fig_dir: str):
    # Save a visual version of the transition matrix for reports and debugging.
    plt.figure(figsize=(5,4))
    ax = plt.gca()
    im = ax.imshow(P, aspect="auto", origin="upper")
    plt.title(f"Gen {gen:02d} Markov Transition (parent bin → child bin)")
    plt.xlabel("Child rank bin")
    plt.ylabel("Parent rank bin")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    path = os.path.join(fig_dir, f"gen_{gen:02d}_transition.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def population_to_df(pop: List[Individual]) -> pd.DataFrame:
    # Convert in memory GA objects into a table that is easy to inspect and export.
    rows = []
    for ind in pop:
        p = ind.params
        rows.append({
            "id": ind.id,
            # store positive RMSE for readability; fitness is -RMSE
            "val_rmse": (-ind.fitness) if ind.fitness is not None else None,
            "units": p["units"],
            "layers": p["layers"],
            "drop": p["dropout"],
            "bs": p["batch_size"],
            "bidir": int(p["bidir"]),
        })
    df = pd.DataFrame(rows).sort_values(by="val_rmse", ascending=True)  # lower is better
    return df

def print_top_table(pop: List[Individual], k: int = 10):
    df = population_to_df(pop).head(k)
    print("Top individuals this generation (by lowest val RMSE):")
    print("rank id    val_rmse  units layers drop  bs  bidir")
    print("--------------------------------------------------")
    for i, row in enumerate(df.itertuples(index=False), 1):
        print(f"{i:<5}{int(row.id):<6}{row.val_rmse:>8.4f}  {int(row.units):<5} "
              f"{int(row.layers):<6} {row.drop:<4} {int(row.bs):<3} {int(row.bidir):<5}")


def params_key(p: Dict) -> Tuple:
    # Turn a parameter dictionary into a hashable tuple so duplicates can be removed.
    return (
        int(p["units"]),
        int(p["layers"]),
        float(p["dropout"]),
        int(p["batch_size"]),
        bool(p["bidir"]),
    )

# ------------------------------- GA Loop --------------------------------------
def evaluate_individual(ind: Individual, X_tr, y_tr, X_val, y_val, epochs=20, verbose=0) -> float:
    # Each individual represents one LSTM hyperparameter configuration.
    m = build_model_from_params(
        ind.params,
        sequence_length=X_tr.shape[1],
        num_features=X_tr.shape[2],
        output_dim=y_tr.shape[1],
    )
    rmse = train_and_eval(
        m, X_tr, y_tr, X_val, y_val,
        epochs=epochs,
        batch_size=ind.params["batch_size"],
        verbose=verbose
    )
    # Fitness is NEGATIVE RMSE so that "larger is better"
    ind.fitness = -rmse
    return -rmse

def run_ga(
    X_tr, y_tr, X_val, y_val,
    pop_size=50, max_gens=30, epochs_per_eval=20,
    elite_frac=0.15, mutation_prob=0.45, crossover_frac=0.7,
    patience=5, min_delta=1e-3,
    verbose_eval=False,
    # --- Markov robustness knobs ---
    markov_alpha: float = 0.1,   
    markov_lazy: bool = False,  
    low_sample_threshold: int = 3,
    num_bins: int = 10,
    finalist_k: int = 3,
):
    # Use a local RNG so GA selection/mutation logic stays tied to the project seed.
    rng = random.Random(GLOBAL_SEED)
    next_id = 0

    # Start from a random population so the GA explores several model shapes immediately.
    population: List[Individual] = []
    for _ in range(pop_size):
        ind = Individual(params=sample_params(), id=next_id)
        next_id += 1
        population.append(ind)

    # Exact global best tracking is deliberately independent of the
    # min_delta threshold used to decide early stopping stagnation.
    global_best_fit = -1e18
    global_best_ind: Optional[Individual] = None
    global_best_generation: Optional[int] = None
    early_stop_reference_fit = -1e18
    stagnation = 0
    generation_records: List[Dict] = []

    print(f"\nStarting GA with {pop_size} individuals for up to {max_gens} generations\n")

    # Score the initial random population before evolution begins.
    for ind in population:
        if ind.fitness is None:
            evaluate_individual(
                ind, X_tr, y_tr, X_val, y_val,
                epochs=epochs_per_eval,
                verbose=0 if not verbose_eval else 2
            )

    # GA generations
    for gen in range(1, max_gens + 1):
        print(f"=== Generation {gen} ===\n")

        # Sort by fitness (higher better -> lower RMSE)
        population.sort(
            key=lambda z: z.fitness if z.fitness is not None else -1e18,
            reverse=True
        )

        # Convert the sorted population into rank bins so the Markov view tracks
        # how parent quality translates into child quality across generations.
        parent_ranks = list(range(len(population)))  # 0 best .. n-1 worst
        parent_bins = rank_bins(parent_ranks, num_bins=num_bins)

        # Keep a small top fraction exactly as they are.
        elite_k = max(1, int(elite_frac * pop_size))
        elites = population[:elite_k]

        # This is fitness proportionate(Roulette wheel) selection: selection probability is proportional
        # to each individual's fitness derived weight.
        # Fitness values are NEGATIVE RMSE. Convert to positive weights for roulette.
        # We shift by the current best so weights are positive and preserve ordering.
        def make_roulette_selector(pop: List[Individual]):
            fits = np.array([float(ind.fitness) for ind in pop], dtype=float)  # higher is better
            # shift to strictly positive
            f_min = float(np.min(fits))
            weights = fits - f_min + 1e-12
            total = float(np.sum(weights))
            if not np.isfinite(total) or total <= 0.0:
                cdf = np.cumsum(np.ones(len(pop)) / len(pop))
            else:
                probs = weights / total
                cdf = np.cumsum(probs)
                cdf[-1] = 1.0
            def select_one():
                # Draw one parent using roulette wheel / fitness proportionate selection.
                r = rng.random()
                idx = int(np.searchsorted(cdf, r, side="right"))
                if idx >= len(pop):
                    idx = len(pop) - 1
                return pop[idx]
            return select_one

        roulette_select = make_roulette_selector(population)
        # ------------------------------------------------------------------------------

        # Build the next generation and record which parent bins contributed to each child.
        children: List[Individual] = []
        created_child_records: List[Tuple[int, int]] = []

        # Elites survive unchanged so the current best solutions are never lost.
        for e in elites:
            child = Individual(params=dict(e.params), id=next_id)
            next_id += 1
            children.append(child)

            pbin = int(parent_bins[population.index(e)])
            created_child_records.append((child.id, pbin))

        # Fill rest by crossover/mutation
        while len(children) < pop_size:
            if rng.random() < crossover_frac:
                # Crossover: TWO parents contribute genetically
                p1 = roulette_select()
                p2 = roulette_select()
                child_params = crossover_params(p1.params, p2.params)
                if rng.random() < mutation_prob:
                    child_params = mutate_params(child_params)

                p1_bin = int(parent_bins[population.index(p1)])
                p2_bin = int(parent_bins[population.index(p2)])

                child = Individual(params=child_params, id=next_id)
                next_id += 1
                children.append(child)

                # Record both parents so the transition matrix reflects crossover ancestry.
                created_child_records.append((child.id, p1_bin))
                created_child_records.append((child.id, p2_bin))

            else:
                # If crossover is skipped, create a child from one parent only.
                p = roulette_select()
                child_params = (
                    mutate_params(p.params)
                    if rng.random() < mutation_prob
                    else dict(p.params)
                )
                p_bin = int(parent_bins[population.index(p)])

                child = Individual(params=child_params, id=next_id)
                next_id += 1
                children.append(child)
                created_child_records.append((child.id, p_bin))

        # Every new child must be trained and scored before ranking the generation.
        for ch in children:
            evaluate_individual(
                ch, X_tr, y_tr, X_val, y_val,
                epochs=epochs_per_eval,
                verbose=0 if not verbose_eval else 2
            )

        # Sort children by fitness to define child rank bins
        children.sort(key=lambda z: z.fitness, reverse=True)
        child_ranks_sorted = list(range(len(children)))
        child_bins_sorted = rank_bins(child_ranks_sorted, num_bins=num_bins)

        # After sorting, we need a quick lookup from child ID to its final quality bin.
        id_to_childbin = {
            children[i].id: int(child_bins_sorted[i]) for i in range(len(children))
        }

        # Convert the recorded ancestry into parallel parent/child bin sequences.
        parent_seq: List[int] = []
        child_seq: List[int] = []
        for (cid, pbin) in created_child_records:
            parent_seq.append(int(pbin))
            child_seq.append(int(id_to_childbin[cid]))

        # The transition matrix summarizes how solution quality moves from parent bins
        # to child bins; lambda2/relaxation time give a compact stability signal.
        P, lambda2, relaxation_time, counts = compute_markov_and_lambda2(
            parent_seq, child_seq,
            num_bins=num_bins,
            alpha=markov_alpha,
            use_lazy=markov_lazy
        )

        # Low sample warning (pre-smoothing counts)
        low_bins = np.where(counts < low_sample_threshold)[0]
        if len(low_bins) > 0:
            low_bin_ids = [int(b) for b in low_bins.tolist()]
            print(
                f"Note: low-sample parent bins this gen (<{low_sample_threshold} samples): "
                f"{low_bin_ids}"
            )

        # Log generation stats (remember: fitness = -rmse)
        fits = [ind.fitness for ind in children]
        rmses = [-f for f in fits]
        gen_best_rmse = float(np.min(rmses))
        gen_mean_rmse = float(np.mean(rmses))

        print("Generation {} Statistics:".format(gen))
        print(f"- Best val RMSE:    {gen_best_rmse:.4f}")
        print(f"- Average val RMSE: {gen_mean_rmse:.4f}")
        print(
            f"- λ₂ (SLEM): {lambda2:.4f}, Relaxation time: "
            f"{relaxation_time:.2f} generations" + (" [lazy]" if markov_lazy else "")
        )
        print("- Transition matrix (rows=parent rank-bin, cols=child rank-bin):")
        print_matrix(P)

        # Pretty table of top individuals
        print_top_table(children, k=10)

        # Persist each generation so the search can be inspected after training finishes.
        df = population_to_df(children)
        df.insert(0, "generation", gen)
        df.insert(1, "rank", np.arange(1, len(df) + 1, dtype=int))
        df["is_generation_best"] = df["rank"].eq(1)
        csv_path = os.path.join(OUT_DIR, f"gen_{gen:02d}_population.csv")
        df.to_csv(csv_path, index=False)
        save_transition_heatmap(P, gen, FIG_DIR)
        print(f"Saved population CSV: {os.path.basename(csv_path)}")
        print(f"Saved transition heatmap: gen_{gen:02d}_transition.png\n")

        # Always retain the exact numerical global best (larger fitness means
        # lower RMSE). min_delta must not affect final model selection.
        current_best_ind = max(children, key=lambda z: z.fitness)
        current_best_fit = float(current_best_ind.fitness)
        if current_best_fit > global_best_fit:
            global_best_fit = current_best_fit
            global_best_ind = copy.deepcopy(current_best_ind)
            global_best_generation = gen

        # min_delta is used only for the early stopping stagnation decision.
        if current_best_fit > early_stop_reference_fit + min_delta:
            early_stop_reference_fit = current_best_fit
            stagnation = 0
        else:
            stagnation += 1

        generation_records.append({
            "generation": gen,
            "best_candidate_id": int(current_best_ind.id),
            "best_val_rmse": gen_best_rmse,
            "mean_val_rmse": gen_mean_rmse,
            "global_best_so_far": float(-global_best_fit),
            "global_best_generation": int(global_best_generation),
            "global_best_candidate_id": int(global_best_ind.id),
            "lambda2_slem": lambda2,
            "relaxation_time": relaxation_time,
        })
        pd.DataFrame(generation_records).to_csv(
            os.path.join(OUT_DIR, "generation_summary.csv"), index=False
        )

        # Mark the individual that established the search wide best at this
        # point. Rewriting here keeps this provenance beside every candidate.
        df["is_global_best_so_far"] = False
        if global_best_generation == gen:
            df.loc[df["id"].eq(global_best_ind.id), "is_global_best_so_far"] = True
        df.to_csv(csv_path, index=False)

        # Next population
        population = children

        if stagnation >= patience:
            print(
                f"Early stopping: <{min_delta} fitness improvement "
                f"for {patience} gens\n"
            )
            break

    # Keep final generation finalists as diagnostics only; they do not override
    # the exact global best found anywhere in the search.
    population.sort(
        key=lambda z: z.fitness if z.fitness is not None else -1e18,
        reverse=True
    )
    finalists: List[Dict] = [dict(ind.params) for ind in population[:max(1, int(finalist_k))]]

    if global_best_ind is None or global_best_generation is None:
        raise RuntimeError("GA completed without evaluating any individual")

    # Return the exact global best, its provenance, and diagnostic finalists.
    return (
        global_best_ind,
        -global_best_fit,
        global_best_generation,
        global_best_ind.id,
        finalists,
    )

# ------------------------------- Main -----------------------------------------
def train_single_stock(args, stock: str, clear_existing: bool = True) -> Dict:
    # This function runs the full training pipeline for one stock ticker from start to finish.
    stock = clean(stock)
    stock, partner_stock = resolve_pair_context_for_stock(
        stock=stock,
        use_streamlit_selection=args.use_streamlit_selection,
        target_stock=args.target,
        partner_stock=args.partner,
        selection_path=args.selection,
    )
    stock_out_dir = os.path.join(ROOT_OUT_DIR, stock)
    configure_output_dirs(stock_out_dir, clear_existing=clear_existing)
    time_step = 120

    # Phase 1: train a fixed baseline so the optimized model has a fair reference point.
    print(f"Preparing data and training baseline LSTM model ({stock})...")
    (
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_te,
        y_te,
        scaler,
        y_te_inv,
        feature_names,
        aligned_stock_df,
    ) = load_data_for_stock(
        stock=stock,
        partner_stock=partner_stock,
        time_step=time_step,
        parquet_path=args.parquet,
    )
    split_summary = build_dataset_split_summary(
        df=aligned_stock_df,
        stock=stock,
        time_step=time_step,
        train_frac=0.8,
        val_frac=0.1,
    )
    split_plot_path = save_dataset_split_plot(aligned_stock_df, stock, split_summary, OUT_DIR)
    split_detail_plot_path = save_dataset_split_detail_plot(aligned_stock_df, stock, split_summary, OUT_DIR)
    print(f"Dataset split summary ({stock}):")
    print(json.dumps(split_summary, indent=2))
    print(f"Using engineered features for {stock}: {feature_names}")
    print(f"Conditioning partner for {stock}: {partner_stock}")
    print(f"Saved dataset split plot: {split_plot_path}")
    print(f"Saved detailed dataset split plot: {split_detail_plot_path}")

    baseline = build_baseline_lstm(
        sequence_length=time_step,
        num_features=X_tr.shape[2],
        output_dim=1,
        units=100,
        dropout=0.2,
        bidir=True,
    )
    hist = baseline.fit(
        # Validation data is supplied during training so early stopping can monitor generalization.
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.baseline_epochs,
        batch_size=64,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                min_delta=1e-4,
            )
        ],
    )
    val_mse = float(np.min(hist.history["val_mse"]))
    val_rmse = math.sqrt(val_mse)
    baseline_learning_curve_path = save_learning_curve_plot(
        hist,
        stock=stock,
        stage_label="Baseline",
        out_dir=OUT_DIR,
    )
    # Predict on the held out test set and convert back to real price units.
    y_pred_te_base = baseline.predict(X_te, verbose=0)
    y_pred_te_base_inv = scaler.inverse_transform(y_pred_te_base).reshape(-1)
    y_te_inv = y_te_inv.reshape(-1)
    rmse_base_test = float(math.sqrt(mean_squared_error(y_te_inv, y_pred_te_base_inv)))

    print(f"\nBaseline Validation RMSE ({stock}): {val_rmse:.4f}")
    print(f"Baseline Test RMSE ({stock}): {rmse_base_test:.4f}")
    print(f"{stock}: baseline={rmse_base_test:.4f} -> final=Pending")

    # Phase 2: search for stronger hyperparameters with the GA.
    print(f"\nInitializing Genetic Algorithm for LSTM optimization ({stock})...\n")
    best_ind, best_ga_valrmse, best_ga_generation, best_ga_candidate_id, _ga_finalists = run_ga(
        X_tr, y_tr, X_val, y_val,
        pop_size=args.pop_size, max_gens=args.max_gens, epochs_per_eval=args.epochs_per_eval,
        elite_frac=0.15, mutation_prob=0.45, crossover_frac=0.7,
        patience=5, min_delta=1e-3,
        verbose_eval=False,
        markov_alpha=0.1,
        markov_lazy=False,
        low_sample_threshold=3,
        num_bins=10,
        finalist_k=args.finalist_k,
    )

    print(f"\n=== Training Final Optimized Model ({stock}) ===")
    print("Best Individual Parameters:")
    for k, v in best_ind.params.items():
        print(f"- {k}: {v}")

    ga_val_rmse = float(best_ga_valrmse)
    winner = {
        "label": "ga_best",
        "selection_method": "exact_global_minimum_search_val_rmse",
        "generation": int(best_ga_generation),
        "candidate_id": int(best_ga_candidate_id),
        "params": dict(best_ind.params),
        "val_rmse_mean": ga_val_rmse,
        "val_rmse_std": 0.0,
        "val_rmse_runs": [ga_val_rmse],
    }
    selected_params = winner["params"]

    print("\nFinal model selection: using the single best GA candidate without top-k reselection.")
    print(
        f"Selected GA best by validation RMSE: {winner['label']} "
        f"(generation={winner['generation']}, candidate_id={winner['candidate_id']}, "
        f"search_val_rmse={winner['val_rmse_mean']:.4f})"
    )

    best_model = build_model_from_params(
        selected_params,
        sequence_length=time_step,
        num_features=X_tr.shape[2],
        output_dim=1,
    )
    final_early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        min_delta=1e-4,
    )
    best_model_history = best_model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.final_epochs,
        batch_size=selected_params["batch_size"],
        verbose=2,
        callbacks=[final_early_stop],
    )
    # This is a new stochastic fit with different training limits from the GA
    # evaluation. Report its restored best weight validation score separately.
    final_eval = best_model.evaluate(X_val, y_val, verbose=0)
    final_retrained_val_mse = float(
        final_eval[0] if isinstance(final_eval, (list, tuple)) else final_eval
    )
    final_retrained_val_rmse = math.sqrt(final_retrained_val_mse)
    final_best_epoch = int(
        getattr(
            final_early_stop,
            "best_epoch",
            int(np.argmin(best_model_history.history["val_mse"])),
        )
        + 1
    )
    final_epochs_ran = int(len(best_model_history.history["val_mse"]))
    final_learning_curve_path = save_learning_curve_plot(
        best_model_history,
        stock=stock,
        stage_label="Final",
        out_dir=OUT_DIR,
    )
    y_pred_tr_final = best_model.predict(X_tr, verbose=0)
    y_pred_val_final = best_model.predict(X_val, verbose=0)
    y_pred_te_final = best_model.predict(X_te, verbose=0)
    y_tr_inv = scaler.inverse_transform(y_tr).reshape(-1)
    y_val_inv = scaler.inverse_transform(y_val).reshape(-1)
    y_pred_tr_final_inv = scaler.inverse_transform(y_pred_tr_final).reshape(-1)
    y_pred_val_final_inv = scaler.inverse_transform(y_pred_val_final).reshape(-1)
    y_pred_te_final_inv = scaler.inverse_transform(y_pred_te_final).reshape(-1)
    rmse_final_test = float(math.sqrt(mean_squared_error(y_te_inv, y_pred_te_final_inv)))
    train_plot_path = save_prediction_phase_plot(
        y_tr_inv,
        y_pred_tr_final_inv,
        stock=stock,
        phase_label="Train",
        out_dir=OUT_DIR,
    )
    val_plot_path = save_prediction_phase_plot(
        y_val_inv,
        y_pred_val_final_inv,
        stock=stock,
        phase_label="Validation",
        out_dir=OUT_DIR,
    )

    print(f"\nFinal retrained Validation RMSE ({stock}): {final_retrained_val_rmse:.4f}")
    print(
        f"Final retraining best epoch: {final_best_epoch} "
        f"(ran {final_epochs_ran}/{args.final_epochs} epochs)"
    )
    print(f"Final Test RMSE ({stock}): {rmse_final_test:.4f}")
    print(
        f"{stock}: baseline={rmse_base_test:.4f} -> final={rmse_final_test:.4f} "
        f"| improvement={(rmse_base_test-rmse_final_test):.4f}"
    )

    # Residuals help us inspect where the model overpredicts or underpredicts.
    errors = y_pred_te_final_inv - y_te_inv
    abs_errors = np.abs(errors)
    rolling_window = int(min(20, len(abs_errors)))
    if rolling_window >= 2:
        rolling_rmse = np.sqrt(
            pd.Series(errors).pow(2).rolling(rolling_window, min_periods=rolling_window).mean()
        ).to_numpy()
    else:
        rolling_rmse = np.array([], dtype=float)

    predictions_path = os.path.join(OUT_DIR, f"predictions_{stock}.csv")
    pred_df = pd.DataFrame({
        "timestep": np.arange(len(y_te_inv), dtype=int),
        "true": y_te_inv,
        "pred": y_pred_te_final_inv,
        "error": errors,
        "abs_error": abs_errors,
    })
    # Saving predictions makes it easy to analyze performance outside the script later.
    pred_df.to_csv(predictions_path, index=False)

    # Save a compact visual summary for the demonstration: full series, zoomed view,
    # residuals, and rolling RMSE.
    zoom_n = int(min(80, len(y_te_inv)))
    plot_path = os.path.join(OUT_DIR, f"prediction_{stock}.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes.ravel()

    ax[0].plot(y_te_inv, label=f"True {stock}", linewidth=1.5)
    ax[0].plot(y_pred_te_final_inv, label=f"Pred {stock}", linewidth=1.2)
    ax[0].set_title(f"Full Test Prediction: {stock}")
    ax[0].set_xlabel("Test timestep")
    ax[0].set_ylabel("Close Price")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(y_te_inv[-zoom_n:], label=f"True {stock}", linewidth=1.5)
    ax[1].plot(y_pred_te_final_inv[-zoom_n:], label=f"Pred {stock}", linewidth=1.2)
    ax[1].set_title(f"Zoom (last {zoom_n} steps)")
    ax[1].set_xlabel("Relative timestep")
    ax[1].set_ylabel("Close Price")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    ax[2].plot(errors, color="tab:red", linewidth=1.2, label="Residual (pred-true)")
    ax[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax[2].set_title("Residuals")
    ax[2].set_xlabel("Test timestep")
    ax[2].set_ylabel("Error")
    ax[2].grid(True, alpha=0.3)
    ax[2].legend()

    if rolling_window >= 2:
        ax[3].plot(rolling_rmse, color="tab:green", linewidth=1.3, label=f"Rolling RMSE ({rolling_window})")
        ax[3].set_title("Rolling RMSE")
        ax[3].set_xlabel("Test timestep")
        ax[3].set_ylabel("RMSE")
        ax[3].grid(True, alpha=0.3)
        ax[3].legend()
    else:
        ax[3].text(0.5, 0.5, "Not enough points for rolling RMSE", ha="center", va="center")
        ax[3].set_axis_off()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # The summary JSON is the handoff format used by Streamlit to display results later.
    summary = {
        "stock": stock,
        "out_dir": OUT_DIR,
        "dataset_split_summary": split_summary,
        "dataset_split_plot_path": split_plot_path,
        "dataset_split_detail_plot_path": split_detail_plot_path,
        "conditioning_partner_stock": partner_stock,
        "feature_names": feature_names,
        "baseline_val_rmse": val_rmse,
        "baseline_learning_curve_path": baseline_learning_curve_path,
        # Legacy scalar fields retained for Streamlit/backward compatibility.
        "ga_best_val_rmse": ga_val_rmse,
        "final_selected_val_rmse": final_retrained_val_rmse,
        "selected_ga_candidate": {
            "label": winner["label"],
            "selection_method": winner["selection_method"],
            "generation": winner["generation"],
            "candidate_id": winner["candidate_id"],
            "search_val_rmse": ga_val_rmse,
            "params": selected_params,
        },
        "ga_evaluation_training": {
            "max_epochs": int(args.epochs_per_eval),
            "early_stopping_patience": 4,
            "early_stopping_min_delta": 1e-4,
        },
        "final_retraining": {
            "max_epochs": int(args.final_epochs),
            "epochs_ran": final_epochs_ran,
            "best_epoch": final_best_epoch,
            "early_stopping_patience": 6,
            "early_stopping_min_delta": 1e-4,
            "validation_rmse": final_retrained_val_rmse,
        },
        "final_learning_curve_path": final_learning_curve_path,
        "train_plot_path": train_plot_path,
        "val_plot_path": val_plot_path,
        "used_params": selected_params,
        "used_ga_candidate": True,
        "selected_candidate_label": winner["label"],
        "selected_candidate_generation": winner["generation"],
        "selected_candidate_id": winner["candidate_id"],
        "selected_candidate_selection_method": winner["selection_method"],
        "selected_candidate_val_rmse_mean": winner["val_rmse_mean"],
        "selected_candidate_val_rmse_std": winner["val_rmse_std"],
        "selected_candidate_val_rmse_runs": winner["val_rmse_runs"],
        "baseline_test_rmse": rmse_base_test,
        "final_test_rmse": rmse_final_test,
        "improvement_abs": (rmse_base_test - rmse_final_test),
        "improvement_pct": ((rmse_base_test - rmse_final_test) / rmse_base_test * 100.0) if rmse_base_test != 0 else 0.0,
        "plot_path": plot_path,
        "predictions_path": predictions_path,
        "summary_path": SUMMARY_PATH,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nArtifacts:")
    print(f"- CSVs per generation: {OUT_DIR}/gen_XX_population.csv")
    print(f"- Generation summary: {OUT_DIR}/generation_summary.csv")
    print(f"- Transition heatmaps: {FIG_DIR}/gen_XX_transition.png")
    print(f"- Dataset split plot: {split_plot_path}")
    print(f"- Detailed dataset split plot: {split_detail_plot_path}")
    if baseline_learning_curve_path:
        print(f"- Baseline learning curve: {baseline_learning_curve_path}")
    if final_learning_curve_path:
        print(f"- Final learning curve: {final_learning_curve_path}")
    print(f"- Train prediction plot: {train_plot_path}")
    print(f"- Validation prediction plot: {val_plot_path}")
    print(f"- Prediction plot: {plot_path}")
    print(f"- Predictions CSV: {predictions_path}")
    print(f"- Summary metrics: {SUMMARY_PATH}")
    print("\nDone.")
    return summary


def main(args):
    configure_output_dirs(ROOT_OUT_DIR, clear_existing=False)

    if args.single_stock:
        # Internal mode used by parallel execution: train just one stock and exit.
        train_single_stock(args, stock=args.single_stock, clear_existing=True)
        return

    # Normal mode trains the target and partner independently, then merges their summaries.
    target, partner = resolve_stock_pair(
        use_streamlit_selection=args.use_streamlit_selection,
        target_stock=args.target,
        partner_stock=args.partner,
        selection_path=args.selection,
    )
    stocks = [target, partner]
    print(f"Running independent training for stocks: {stocks}")

    stock_summaries: Dict[str, Dict] = {}
    if args.parallel:
        # Parallel mode launches one child process per stock to reduce wall clock time.
        print("Parallel mode enabled: launching one process per stock.")
        procs = []
        for s in stocks:
            # Spawn this same script again, but force it into single stock mode.
            cmd = [
                "python", "-u", __file__,
                "--parquet", args.parquet,
                "--selection", args.selection,
                "--target", args.target,
                "--partner", args.partner,
                "--pop-size", str(args.pop_size),
                "--max-gens", str(args.max_gens),
                "--epochs-per-eval", str(args.epochs_per_eval),
                "--baseline-epochs", str(args.baseline_epochs),
                "--final-epochs", str(args.final_epochs),
                "--single-stock", s,
            ]
            procs.append((s, subprocess.Popen(cmd)))
        for s, p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"Training failed for {s} with exit code {rc}")
    else:
        for s in stocks:
            train_single_stock(args, stock=s, clear_existing=True)

    for s in stocks:
        # Each stock subprocess writes its own summary, which we combine here.
        sp = os.path.join(ROOT_OUT_DIR, s, "summary.json")
        if not os.path.exists(sp):
            raise FileNotFoundError(f"Expected summary not found: {sp}")
        with open(sp, "r", encoding="utf-8") as f:
            stock_summaries[s] = json.load(f)

    combined_summary = {
        "mode": "parallel" if args.parallel else "sequential",
        "stocks": stocks,
        "per_stock": stock_summaries,
        "timestamp": int(time.time()),
    }
    combined_summary_path = os.path.join(ROOT_OUT_DIR, "summary.json")
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)
    print(f"\nCombined summary written to {combined_summary_path}")

if __name__ == "__main__":
    # Command line flags let the same script support Streamlit mode, manual mode,
    # sequential runs, and internal per stock subprocess runs.
    parser = argparse.ArgumentParser(description="Train GA+Markov LSTM independently for two selected stocks.")
    parser.add_argument("--parquet", default=PARQUET_PATH, help="Path to parquet close-price dataset.")
    parser.add_argument("--selection", default=SELECTION_PATH, help="Path to selected_pair.json from Streamlit.")
    parser.add_argument("--use-streamlit-selection", dest="use_streamlit_selection", action="store_true", default=True)
    parser.add_argument("--no-streamlit-selection", dest="use_streamlit_selection", action="store_false")
    parser.add_argument("--target", default=TARGET_STOCK, help="Manual target ticker when not using Streamlit selection.")
    parser.add_argument("--partner", default=PARTNER_STOCK, help="Manual partner ticker when not using Streamlit selection.")
    parser.add_argument("--parallel", action="store_true", help="Run the two stock trainings in parallel (two processes).")
    parser.add_argument("--single-stock", default="", help="Internal mode: run training only for one stock ticker.")
    parser.add_argument("--pop-size", type=int, default=30, help="GA population size.")
    parser.add_argument("--max-gens", type=int, default=20, help="Maximum GA generations.")
    parser.add_argument("--epochs-per-eval", type=int, default=10, help="Training epochs for each GA individual evaluation.")
    parser.add_argument("--baseline-epochs", type=int, default=20, help="Epochs for baseline model fit.")
    parser.add_argument("--final-epochs", type=int, default=40, help="Epochs for final selected model fit.")
    parser.add_argument("--finalist-k", type=int, default=3, help="Top-k GA finalists to re-train for final selection.")
    parser.add_argument("--finalist-seeds", type=int, default=3, help="Number of random-seed repeats per finalist during validation-based selection.")
    parsed, _unknown = parser.parse_known_args()
    main(parsed)
