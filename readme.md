# Topic Classification

### Made by Aryan Kumar, 24110055

## Overview

This project implements a scalable pipeline to classify text into predefined topics using a large dataset (~4GB, 10 million rows). The focus is on efficient data handling, strong baseline models, and reproducibility.

---

## Project Structure

```bash
project/
│── src/
│   ├── train.py
│   ├── inference.py
│   ├── model.py
│   └── utils.py
│
│── experiments/
│── final_models/
│
│── report.pdf
│── requirements.txt
│── README.md
```

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone <repo-link>
cd project
```

### 2. Create Environment

```bash
python -m venv venv

# Activate
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

* Format: Parquet
* Size: ~4GB
* Rows: ~10 million
* Columns:

  * `DATA`: Input text
  * `TOPIC`: Label

⚠️ The dataset is processed using chunking/lazy loading to avoid memory issues.

---

## Training

Run:

```bash
python src/train.py
```

What it does:

* Loads data in chunks
* Preprocesses text
* Converts text → TF-IDF features
* Trains model
* Saves trained model to `final_models/`

---

## Inference

Run:

```bash
python src/inference.py --input "Your text here"
```

Output:

```bash
Predicted Topic: <label>
```

---

## Input / Output Schema

### Input

```json
{
  "DATA": "Sample input text"
}
```

### Output

```json
{
  "TOPIC": "Predicted label"
}
```

---

## Dependencies

Main libraries used:

* numpy
* pandas
* scikit-learn
* pyarrow / dask

Install via:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

* Fixed random seeds
* Deterministic pipeline
* End-to-end execution supported

---
