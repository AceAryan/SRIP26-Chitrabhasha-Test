# Topic Classification — SRIP 2026

Multi-class text topic classifier trained from scratch on 10M rows across 24 topic categories.
Final model: **TF-IDF + Logistic Regression** (83–89% accuracy depending on data size).

## Project Structure

```
project/
├── src/
│   ├── train.py              # TF-IDF + LogReg training pipeline (final model)
│   └── inference.py          # Inference: single text, top-k, or batch CSV
├── experiments/              # All experimental scripts
│   ├── exp1_tfidf_logreg.py  # Experiment 1 — baseline (5K per class)
│   ├── exp2_fasttext_v1.py   # Experiment 2a — FastText V1
│   ├── exp2_fasttext_v2.py   # Experiment 2b — FastText V2 (improved)
│   ├── exp3_linear_svm.py    # Experiment 3 — Linear SVM
│   └── exp4_textcnn.py       # Experiment 4 — TextCNN (deep learning)
├── final_models/
│   └── tfidf_logreg_final.pkl  # Saved model bundle (vectorizer + classifier)
├── report.pdf
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Environment

Python 3.10+ recommended. Create a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Note: PyTorch is only required for the experimental deep learning scripts (exp2, exp4).
> The final model (train.py + inference.py) runs on CPU with no GPU required.

---

## Training Instructions

### Quick Start

```bash
python src/train.py --data path/to/dataset_10M.parquet
```

### Full Command with Options

```bash
python src/train.py \
  --data path/to/dataset_10M.parquet \
  --max_features 150000 \
  --C 5.0 \
  --output final_models/tfidf_logreg_final.pkl
```

### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--data` | required | Path to dataset_10M.parquet |
| `--max_features` | 150000 | TF-IDF vocabulary size |
| `--min_df` | 5 | Min document frequency for TF-IDF terms |
| `--C` | 5.0 | Logistic Regression regularization strength |
| `--test_size` | 0.2 | Fraction held out for testing (default = 2M rows) |
| `--chunk_size` | 200000 | Rows read per chunk — reduce if you run out of RAM |
| `--seed` | 42 | Random seed for reproducibility |
| `--output` | final_models/tfidf_logreg_final.pkl | Path to save model bundle |

### Memory-Efficient Chunk Loading

The dataset is 4 GB and cannot be loaded into RAM all at once. The training script
reads the parquet file in chunks of `chunk_size` rows at a time, processes each chunk,
and concatenates only the lightweight text/label lists — never the full Arrow buffers.

Peak RAM usage by stage:

| Stage | Approximate RAM |
|---|---|
| Chunk loading (200K rows at a time) | ~1–2 GB |
| Full text list in memory (10M rows) | ~4–6 GB |
| TF-IDF matrix float32 (8M x 150K sparse) | ~3–4 GB |
| Logistic Regression training | ~2–3 GB additional |

**Total recommended RAM: 16 GB minimum.**

If you run out of memory, reduce `--chunk_size` to limit total rows loaded:

```bash
# Use ~5M rows instead of 10M (safer on 8GB RAM machines)
python src/train.py \
  --data path/to/dataset_10M.parquet \
  --chunk_size 100000
```

### What the training script does

1. Reads the parquet file in chunks of `chunk_size` rows (default 200K)
2. Drops null and empty text rows
3. Label-encodes the 24 TOPIC classes
4. Performs stratified 80/20 train/test split on the full loaded data
5. Fits TF-IDF vectorizer on training set only — no leakage from test set
6. Uses float32 sparse matrix to halve RAM vs default float64
7. Trains Logistic Regression with saga solver across all CPU cores
8. Prints per-class precision, recall, F1 on 2M test rows
9. Saves vectorizer + classifier + label encoder as a single .pkl bundle
10. Runs a sanity check on 6 sample texts

### Expected Training Time

| Stage | Time |
|---|---|
| Loading 10M rows in chunks | ~5–10 min |
| TF-IDF fit + transform | ~10–20 min |
| Logistic Regression training | ~20–60 min |
| **Total** | **~35–90 min (CPU only)** |

---

## Inference Instructions

### Single Text

```bash
python src/inference.py \
  --model final_models/tfidf_logreg_final.pkl \
  --text "The stock market rallied today after the Fed announced a rate cut"
```

Output:
```
Predicted topic : finance_and_business
Confidence      : 0.9341
```

### Top-K Predictions

```bash
python src/inference.py \
  --model final_models/tfidf_logreg_final.pkl \
  --text "Arsenal beat Manchester City 2-1 in the Premier League" \
  --topk 3
```

Output:
```
Top predictions:
  sports_and_fitness                  0.9512
  entertainment                       0.0231
  politics                            0.0087
```

### Batch CSV Inference

Input CSV must have a DATA column:

```bash
python src/inference.py \
  --model final_models/tfidf_logreg_final.pkl \
  --file input.csv \
  --output predictions.csv
```

---

## Input / Output Schema

### Input

| Field | Type | Description |
|---|---|---|
| `DATA` | string | Raw text to classify (any length) |

### Output

| Field | Type | Description |
|---|---|---|
| `PREDICTED_TOPIC` | string | Predicted class label |
| `CONFIDENCE` | float | Predicted class probability (0-1) |

### Supported Topic Labels

```
adult_content, art_and_design, crime_and_law, education_and_jobs,
electronics_and_hardare, entertainment, fashion_and_beauty,
finance_and_business, food_and_dining, games, health,
history_and_geography, home_and_hobbies, industrial, literature,
politics, religion, science_math_and_technology, social_life,
software, software_development, sports_and_fitness,
transportation, travel_and_tourism
```

---

## Experiment Summary

| Experiment | Model | Data | Accuracy |
|---|---|---|---|
| exp1 | TF-IDF + Logistic Regression | 120K (5K/class) | 83.01% |
| exp2a | FastText V1 | 120K | 68.27% |
| exp2b | FastText V2 | 480K (20K/class) | TBD |
| exp3 | TF-IDF + Linear SVM | 120K | TBD |
| exp4 | TextCNN (deep learning) | 480K | TBD |
| **final** | **TF-IDF + LogReg** | **Full 10M rows** | **TBD** |

---

## Reproducibility

- Random seed fixed at 42 for Python, NumPy, and scikit-learn
- Stratified 80/20 train/test split with fixed seed
- TF-IDF vectorizer fitted on training data only — no leakage from test set
- Model bundle saves vectorizer + classifier + label encoder together
- Chunk-based parquet loading with fixed seed ensures consistent behaviour across runs

---

## Requirements

See `requirements.txt`. Core dependencies for final model:

```
pandas>=2.0.0
pyarrow>=12.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

For deep learning experiments only:

```
torch>=2.0.0
```