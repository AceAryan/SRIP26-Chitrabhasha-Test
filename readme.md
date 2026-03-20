# Topic Classification 
Multi-class text topic classifier trained from scratch on 10M rows across 24 topic categories.

## Project Structure

```
project/
├── src/
│   ├── train.py              # Full training pipeline
│   ├── inference.py          # Inference: single text, batch, or CSV
│   ├── model.py              # TextCNN architecture
│   └── utils.py              # Data loading, vocabulary, preprocessing
├── experiments/              # Logs, configs, checkpoints per experiment
│   ├── exp1_tfidf_logreg.py
│   ├── exp2_fasttext_v1.py
│   ├── exp2_fasttext_v2.py
│   ├── exp3_linear_svm.py
│   └── exp4_textcnn.py
├── final_models/
│   └── textcnn_best.pt       # Best model checkpoint
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

### 3. GPU Support (recommended)

If you have an NVIDIA GPU, install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is available:

```python
import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))
```

---

## Training Instructions

### Quick Start (subset, for experimentation)

```bash
python src/train.py \
  --data path/to/dataset_10M.parquet \
  --n_per_class 20000 \
  --epochs 20 \
  --batch_size 256 \
  --output final_models/textcnn_best.pt
```

### Full Dataset Training

```bash
python src/train.py \
  --data path/to/dataset_10M.parquet \
  --n_per_class 100000 \
  --epochs 15 \
  --batch_size 256 \
  --output final_models/textcnn_best.pt
```

### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--n_per_class` | 20000 | Rows per class to load |
| `--epochs` | 20 | Max training epochs |
| `--batch_size` | 256 | Training batch size |
| `--embed_dim` | 128 | Embedding dimension |
| `--num_filters` | 128 | Conv filters per kernel size |
| `--max_seq_len` | 256 | Max token length per document |
| `--lr` | 1e-3 | Initial learning rate |
| `--seed` | 42 | Random seed |

---

## Inference Instructions

### Single Text

```bash
python src/inference.py \
  --model final_models/textcnn_best.pt \
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
  --model final_models/textcnn_best.pt \
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

```bash
python src/inference.py \
  --model final_models/textcnn_best.pt \
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
| `CONFIDENCE` | float | Softmax probability of predicted class (0–1) |

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

## Reproducibility

- All random seeds fixed at 42 (Python, NumPy, PyTorch)
- Stratified train/test split (80/20)
- Vocabulary built from training set only (no data leakage)
- Model checkpoint saves best validation accuracy

---

## Requirements

See `requirements.txt`. Core dependencies:

```
torch>=2.0.0
pandas>=2.0.0
pyarrow>=12.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
reportlab>=4.0.0
```
