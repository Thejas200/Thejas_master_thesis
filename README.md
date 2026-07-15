# Stock Prediction And Fashion MNIST

This repository contains two GPU based deep learning workflows:

- Stock prediction using a Streamlit stock-pair selector and a GA+Markov LSTM training pipeline.
- Fashion MNIST classification using the GA+Markov CNN training pipeline.

## Important GPU Requirement

This code is intended to run only on a machine with GPU support.

Before starting either workflow, make sure the machine has:

- A supported NVIDIA GPU.
- A working NVIDIA driver.
- Conda, provided by Miniconda or Anaconda.
- Python installed in the version range required by the workflow.

The GPU setup below is intended for Linux or Windows through WSL2. Current
TensorFlow releases do not support GPU execution on native Windows, and macOS
does not have official TensorFlow CUDA support. The `tensorflow[and-cuda]`
installation used below supplies the required CUDA and cuDNN Python libraries,
but the host machine must still have a working NVIDIA driver.

Check the driver before creating an environment:

```bash
nvidia-smi
```

If `conda` is installed but `conda activate` is unavailable in Bash, initialize
it once and then reopen the terminal:

```bash
conda init bash
```

### Anaconda Terms Of Service

Miniconda and Anaconda normally use Anaconda's `main` and `r` default channels.
Recent Conda installations prompt for any required Terms of Service acceptance
automatically when a command such as `conda create` first accesses those
channels. Therefore, the following commands do not have to be run in advance on
every machine.

If Conda stops and explicitly asks for channel acceptance, first review the
terms with `conda tos` or `conda tos view`. If you agree to them, record the
acceptance for the requested channels:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Acceptance is normally recorded for later Conda commands. Conda may ask again
if the terms change. See Anaconda's
[Terms of Service plugin documentation](https://www.anaconda.com/docs/getting-started/tos-plugin)
for details. Organizational users should also check whether their use of
Anaconda's default channels requires an appropriate license.

After installing dependencies, you can check whether TensorFlow can see the GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The command should print at least one GPU device. If it prints an empty list, fix the GPU/TensorFlow/CUDA setup before running the experiments.

## Repository Files

Key files used by the workflows:

- `app.py`: Streamlit app for choosing a target stock and correlated partner stock.
- `train_two_stocks.py`: GA+Markov LSTM training pipeline for the selected stock pair.
- `sp500_close_stooq.parquet`: standard stock close-price dataset used by the app and trainer.
- `selected_pair.json`: stock pair saved by the Streamlit app.
- `requirements.txt`: dependencies for the stock workflow.
- `fashion_mnist.ipynb`: final Fashion MNIST notebook.
- `fashion_mnist.txt`: dependencies for the Fashion MNIST notebook.

## Process 1: Run The Stock Workflow

### 1. Create The Stock Environment

The stock workflow was tested with Python `3.12`.

Create a dedicated Conda environment and install the stock dependencies:

```bash
conda create --name stock-gpu python=3.12 pip -y
conda activate stock-gpu
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "tensorflow[and-cuda]"
```

Creating an environment is a one-time step on each machine. In later terminal
sessions, activate the existing environment before running the stock workflow:

```bash
conda activate stock-gpu
```

### 2. Confirm GPU Access

Run this inside the activated stock environment:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Continue only if TensorFlow lists a GPU.

### 3. Confirm The Stock Dataset

Keep the included standard dataset in the project folder with this filename:

```text
sp500_close_stooq.parquet
```

Both `app.py` and `train_two_stocks.py` use this file by default.

### 4. Choose How To Run The Stock Workflow

After the stock environment, GPU check, and dataset are ready, use one of these two options.

### Option A: Run Everything Through Streamlit (Preferred)

Use this option if you want to select the stock pair and start training from the browser.

Start the Streamlit app:

```bash
streamlit run app.py
```

In the Streamlit browser page:

1. Select the target stock.
2. Choose the number of top correlated stocks to display.
3. Review the ranked correlated partner stocks.
4. Select one partner stock from the list.
5. Click **Save selected pair for next step**.

This creates or updates:

```text
selected_pair.json
```

Then, in the same Streamlit app:

1. Keep **Run stocks in parallel** enabled if the GPU has enough memory.
2. Click **Start Training**.
3. Use **Refresh Log** to monitor progress.
4. Use **Stop Training** if you need to stop the background training process.

Training writes progress and outputs to:

```text
.streamlit_training.log
ga_markov_outputs_lstm/
```

### Option B: Run Stock Training Directly

Use this option if you already have `selected_pair.json` or want to run the trainer from the terminal.

If `selected_pair.json` already exists, run:

```bash
python train_two_stocks.py --parquet sp500_close_stooq.parquet --selection selected_pair.json --use-streamlit-selection --parallel
```

If GPU memory is limited, remove `--parallel`:

```bash
python train_two_stocks.py --parquet sp500_close_stooq.parquet --selection selected_pair.json --use-streamlit-selection
```

To choose tickers manually instead of using `selected_pair.json`:

```bash
python train_two_stocks.py --no-streamlit-selection --target AAPL --partner MSFT
```

### 5. Review Stock Outputs

The stock training pipeline saves model summaries, generated populations, plots, and predictions under:

```text
ga_markov_outputs_lstm/
```

For per-stock runs, check the ticker subfolders inside that directory.

## Process 2: Run The Fashion MNIST Workflow

Use a separate environment for Fashion MNIST so the notebook setup stays isolated from the stock workflow.

Recommended Python version:

- Python `3.11`

### 1. Create The Fashion MNIST Environment

The final notebook records `tf-gpu` as its kernel name. That name does not
create or install the environment automatically. Create it once with:

```bash
conda create --name tf-gpu python=3.11 pip -y
conda activate tf-gpu
python -m pip install --upgrade pip
python -m pip install -r fashion_mnist.txt
python -m pip install "tensorflow[and-cuda]"
python -m pip install ipykernel
python -m ipykernel install --user --name tf-gpu --display-name "Python (tf-gpu)"
```

If `tf-gpu` already appears in `conda env list`, do not create it again. Activate
it and run the installation commands beginning with the pip upgrade. In later
terminal sessions, only activation is required:

```bash
conda activate tf-gpu
```

### 2. Confirm GPU Access

Run this inside the activated Fashion MNIST environment:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Continue only if TensorFlow lists a GPU.

### 3. Run The Notebook

Open the following file directly in VS Code or another notebook-capable editor:

```text
fashion_mnist.ipynb
```

Select `Python (tf-gpu)` as the notebook kernel, then run the cells from top to
bottom. The standalone Jupyter application is not required. `ipykernel` is
installed only to let the editor execute notebook cells in the Conda
environment.

## Common Commands

List the Conda environments available on the current machine:

```bash
conda env list
```

Deactivate the current Conda environment:

```bash
conda deactivate
```

Reinstall stock dependencies:

```bash
conda activate stock-gpu
python -m pip install -r requirements.txt
python -m pip install "tensorflow[and-cuda]"
```

Reinstall Fashion MNIST dependencies:

```bash
conda activate tf-gpu
python -m pip install -r fashion_mnist.txt
python -m pip install "tensorflow[and-cuda]"
```

Each environment is isolated. Installing packages in `tf-gpu` does not install
them in `stock-gpu` or any other environment. A new environment must be created
and have its dependencies installed once before it can run either workflow.

For TensorFlow's current platform and GPU installation requirements, see the
[official TensorFlow pip installation guide](https://www.tensorflow.org/install/pip).
