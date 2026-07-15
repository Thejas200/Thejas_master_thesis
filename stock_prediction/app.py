# app.py
import json
import os
import signal
import subprocess
import numpy as np
import pandas as pd
import streamlit as st

# Number of days used when checking how stable the correlation is over time.
ROLL_WIN = 60
# Ignore pairs that are almost identical.
MAX_FULL_CORR = 0.98
# Default number of candidate partners shown in the UI table.
TOP_N_CORRELATED = 20
SELECTION_PATH = "selected_pair.json"
TRAIN_SCRIPT = "train_two_stocks.py"
TRAIN_OUT_DIR = "ga_markov_outputs_lstm"
TRAIN_SUMMARY_PATH = os.path.join(TRAIN_OUT_DIR, "summary.json")
TRAIN_LOG_PATH = ".streamlit_training.log"
TRAIN_PID_PATH = ".streamlit_training.pid"

# These are near duplicate share classes, so we exclude them from partner suggestions.
EXCLUDE_PAIRS = {("GOOG", "GOOGL"), ("GOOGL", "GOOG"), ("FOX", "FOXA"), ("FOXA", "FOX")}


def clean(t: str) -> str:
    # Normalize ticker strings so lookups behave consistently across files.
    return str(t).strip().upper().replace(".", "-")


@st.cache_data(show_spinner=False)
def load_prices_local(path: str) -> pd.DataFrame:
    # Cache the parquet load so widget interaction does not repeatedly hit disk.
    df = pd.read_parquet(path)
    df.columns = [clean(c) for c in df.columns]
    return df.dropna(axis=0, how="any")


def rolling_corr_score(returns: pd.DataFrame, a: str, b: str, win: int):
    # Compute a rolling Pearson correlation and summarize it with the mean absolute value.
    # This rewards pairs that stay correlated over time, not just on average once.
    rc = returns[a].rolling(win).corr(returns[b]).dropna()
    return float(rc.abs().mean())


def ranked_partners(returns: pd.DataFrame, target: str) -> pd.DataFrame:
    # Build a ranked table of possible partner stocks for the selected target.
    t = clean(target)
    rows = []
    for other in returns.columns:
        if other == t or (t, other) in EXCLUDE_PAIRS:
            continue
        # Filter near duplicate share classes, then rank the rest by correlation strength
        # and rolling correlation consistency.
        # `Series.corr(...)` here is Pearson correlation on the two return series.
        full_corr = float(returns[t].corr(returns[other]))
        if abs(full_corr) > MAX_FULL_CORR:
            continue
        roll_score = rolling_corr_score(returns, t, other, ROLL_WIN)
        rows.append((other, full_corr, abs(full_corr), roll_score))
    df = pd.DataFrame(rows, columns=["partner", "full_corr", "abs_full_corr", "mean_abs_rollcorr_60d"])
    return df.sort_values(["abs_full_corr", "mean_abs_rollcorr_60d"], ascending=False)


def _read_pid(path: str) -> int:
    # Read the training process ID if one was saved earlier.
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return -1


def _is_pid_running(pid: int) -> bool:
    # `os.kill(pid, 0)` does not stop the process; it only checks whether it exists.
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _tail_text(path: str, max_lines: int = 500) -> str:
    # Show only the last part of the log so the page stays readable.
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def _stop_pid(pid: int) -> bool:
    # Training is launched in its own session, so kill the whole process group.
    if pid <= 0:
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


st.set_page_config(layout="wide")
st.title("S&P 500 Correlated Stock Selector")
st.caption("Steps: choose target -> view top correlated stocks -> choose one partner.")

# The included parquet file is the standard dataset for the stock workflow.
path = st.sidebar.text_input("Path to parquet data", "sp500_close_stooq.parquet")

try:
    prices = load_prices_local(path)
except Exception as e:
    st.error(f"Could not load '{path}'. Make sure the standard dataset is in the project folder.\n\n{e}")
    st.stop()

# Use log returns for correlation analysis because returns are more comparable than raw prices.
returns = np.log(prices / prices.shift(1)).dropna()
st.success(f"Loaded dataset: {prices.shape[0]} dates x {prices.shape[1]} tickers")

# Step 1: choose a target ticker, then show the strongest non trivial partners.
target = st.selectbox("1) Select target stock", sorted(returns.columns))
n_top = st.number_input("2) Number of top correlated stocks", min_value=5, max_value=50, value=TOP_N_CORRELATED, step=1)

partners = ranked_partners(returns, target)
if partners.empty:
    st.error("No correlated partners found for this target.")
    st.stop()

# Keep only the top N candidates requested by the user.
top_partners = partners.head(int(n_top)).copy()
top_partners.insert(0, "rank", np.arange(1, len(top_partners) + 1))
st.subheader(f"3) Top {int(n_top)} correlated stocks for {clean(target)}")
st.dataframe(top_partners, use_container_width=True, hide_index=True)

# The selectbox uses the ranked table, so the first option is the strongest candidate.
partner = st.selectbox("4) Select one correlated stock", top_partners["partner"].tolist(), index=0)
choice = top_partners[top_partners["partner"] == partner].iloc[0]
st.success(
    f"Selected pair: {clean(target)} + {partner} | "
    f"full_corr={choice['full_corr']:.4f}, abs_full_corr={choice['abs_full_corr']:.4f}"
)

if st.button("Save selected pair for next step", type="primary"):
    # Persist the choice so the training script can run without re-selecting inside Streamlit.
    payload = {
        "target": clean(target),
        "partner": partner,
        "full_corr": float(choice["full_corr"]),
        "abs_full_corr": float(choice["abs_full_corr"]),
        "mean_abs_rollcorr_60d": float(choice["mean_abs_rollcorr_60d"]),
        "dataset_path": path,
    }
    with open(SELECTION_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    st.info(f"Saved selection to {SELECTION_PATH}")

st.divider()
st.subheader("Train In Streamlit")
st.caption("This runs the full GA+Markov two-stock training pipeline from `train_two_stocks.py`.")
run_parallel = st.checkbox("Run stocks in parallel (2 processes)", value=True)
auto_refresh = st.checkbox("Auto-refresh log while training", value=False)

# If a PID is saved and still active, show that training is already in progress.
existing_pid = _read_pid(TRAIN_PID_PATH)
is_running = _is_pid_running(existing_pid)
if is_running:
    st.info(f"Training running in background (PID {existing_pid}).")
else:
    if existing_pid > 0 and os.path.exists(TRAIN_PID_PATH):
        os.remove(TRAIN_PID_PATH)
    st.caption("No active training process.")

col1, col2, col3 = st.columns(3)
start_clicked = col1.button("Start Training", type="primary")
refresh_clicked = col2.button("Refresh Log")
stop_clicked = col3.button("Stop Training", disabled=not is_running)
if refresh_clicked:
    # Streamlit reruns the script from the top, which refreshes the log and status panels.
    st.rerun()

if stop_clicked:
    pid = _read_pid(TRAIN_PID_PATH)
    if _stop_pid(pid):
        st.warning(f"Sent stop signal to training process group {pid}.")
    else:
        st.warning("No active training process was found to stop.")
    if os.path.exists(TRAIN_PID_PATH):
        os.remove(TRAIN_PID_PATH)
    st.rerun()

if start_clicked:
    if _is_pid_running(_read_pid(TRAIN_PID_PATH)):
        st.warning("Training is already running. Use Refresh Log.")
        st.stop()

    if not os.path.exists(SELECTION_PATH):
        st.error(f"{SELECTION_PATH} not found. Save selected pair first.")
        st.stop()
    if not os.path.exists(TRAIN_SCRIPT):
        st.error(f"{TRAIN_SCRIPT} not found.")
        st.stop()

    # Launch the trainer as a background process and stream its stdout into a log file
    # so the UI remains responsive while training is running.
    cmd = [
        "python",
        "-u",
        TRAIN_SCRIPT,
        "--parquet", path,
        "--selection", SELECTION_PATH,
        "--use-streamlit-selection",
    ]
    if run_parallel:
        cmd.append("--parallel")
    st.write(f"Running: `{' '.join(cmd)}`")
    with open(TRAIN_LOG_PATH, "w", encoding="utf-8") as lf:
        lf.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    with open(TRAIN_PID_PATH, "w", encoding="utf-8") as pf:
        # Persisting the PID lets later reruns know whether training is still active.
        pf.write(str(proc.pid))
    st.success(f"Training started in background (PID {proc.pid}). Click Refresh Log.")
    st.rerun()

st.subheader("Training Log")
log_text = _tail_text(TRAIN_LOG_PATH, max_lines=500)
if log_text:
    # Render the log as plain text so progress messages are easy to inspect.
    st.code(log_text, language="text")
else:
    st.caption("No training log yet.")

if is_running and auto_refresh:
    st.caption("Auto-refresh is enabled. The app will rerun every 10 seconds while training is running.")
    st.info("If the page feels jumpy, turn off auto-refresh and use Refresh Log instead.")

if (not _is_pid_running(_read_pid(TRAIN_PID_PATH))) and os.path.exists(TRAIN_SUMMARY_PATH):
    # Once the process finishes, expose the saved metrics and generated plots in the UI.
    with open(TRAIN_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    st.subheader("Summary Metrics")
    st.json(summary)
    if "per_stock" in summary:
        for stock, ss in summary["per_stock"].items():
            st.markdown(f"**{stock}**")
            split_summary = ss.get("dataset_split_summary")
            if split_summary:
                st.caption("Dataset split summary")
                st.json(split_summary)
            for extra_key in [
                "dataset_split_plot_path",
                "dataset_split_detail_plot_path",
                "baseline_learning_curve_path",
                "final_learning_curve_path",
                "train_plot_path",
                "val_plot_path",
            ]:
                extra_path = ss.get(extra_key)
                if extra_path and os.path.exists(extra_path):
                    st.image(extra_path, caption=os.path.basename(extra_path), use_container_width=True)
            p = ss.get("plot_path")
            if p and os.path.exists(p):
                st.image(p, caption=os.path.basename(p), use_container_width=True)
    else:
        split_summary = summary.get("dataset_split_summary")
        if split_summary:
            st.caption("Dataset split summary")
            st.json(split_summary)
        for extra_key in [
            "dataset_split_plot_path",
            "dataset_split_detail_plot_path",
            "baseline_learning_curve_path",
            "final_learning_curve_path",
            "train_plot_path",
            "val_plot_path",
        ]:
            extra_path = summary.get(extra_key)
            if extra_path and os.path.exists(extra_path):
                st.image(extra_path, caption=os.path.basename(extra_path), use_container_width=True)
        for key in ["plot_target_path", "plot_partner_path", "plot_path"]:
            p = summary.get(key)
            if p and os.path.exists(p):
                st.image(p, caption=os.path.basename(p), use_container_width=True)
