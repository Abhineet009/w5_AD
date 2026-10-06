# TECH 405 RNN Hackathon: LSTM on sklearn Digits

An LSTM classifier that reads each 8x8 handwritten digit image **row by row**, so every image is a sequence of 8 time steps with 8 features per step. Built with PyTorch on the built-in scikit-learn Digits dataset (no download needed). No pretrained models.

## Task summary

| Item | Value |
|---|---|
| Dataset | `sklearn.datasets.load_digits` (1,797 images, 10 classes) |
| Sequence view | 8 time steps (rows) x 8 features (columns) |
| Model | `nn.LSTM(8, 64, num_layers=2, dropout=0.2)` then `nn.Linear(64, 10)` |
| Split | Fixed stratified 80/20 train/test (seed 42); 10% of train held out for validation |
| Baseline / target | ~10% (majority class) / 93% test accuracy |
| Result (original `train.py` run) | **97.78%** test accuracy (352 / 360 correct) |

The test set is used once, at the end. The best epoch is chosen on the validation set only.

## Project structure

```
rnn-digits-lstm/
├── README.md                  this file
├── REPORT.md                  full write-up with the four snapshots
├── requirements.txt           Python dependencies
├── train.py                   sectioned training script (writes results/)
├── week5_lstm_digits.ipynb    notebook version (run top to bottom)
├── notebook.py                same walkthrough in percent-cell format
└── results/                   snapshots, loss curve, confusion matrix, metrics.json
```

## Setup

Python 3.9 or newer is recommended.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

**Option A: script**

```
python train.py
```

Prints the data shapes, model, training log and final test accuracy, and saves figures and `metrics.json` into `results/`.

**Option B: notebook**

```
jupyter notebook week5_lstm_digits.ipynb
```

Run all cells top to bottom. Take your report screenshots from the cell outputs (data and shapes, model definition, training log, test accuracy and confusion matrix).

## Notes

- Results are seeded (`SEED = 42`), but exact numbers can vary slightly between machines and PyTorch versions.
- The 97.78% figure comes from the first run of `train.py`. The notebook uses `DataLoader` batching, so its accuracy may differ slightly; report the number from your own run.
- The test set has 360 images, so a single run is accurate to roughly one to two percentage points.

## Report

See [REPORT.md](REPORT.md) for the dataset description, model, training curves, test results and a short reflection.
