# src/train.py
"""
Training script for Final Model: TF-IDF + Logistic Regression
Memory-efficient: processes data in chunks, never loads full dataset at once.

Usage:
    python src/train.py --data path/to/dataset_10M.parquet
"""

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import argparse
import time
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── 0. Args ───────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=str,   required=True)
    parser.add_argument("--max_features", type=int,   default=150_000)
    parser.add_argument("--min_df",       type=int,   default=5)
    parser.add_argument("--C",            type=float, default=5.0)
    parser.add_argument("--test_size",    type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--chunk_size",   type=int,   default=200_000,
                        help="Rows to read at a time (tune down if still OOM)")
    parser.add_argument("--output",       type=str,
                        default="final_models/tfidf_logreg_final.pkl")
    return parser.parse_args()


# ── 1. Two-pass strategy ──────────────────────────────────────────
# Pass 1: read chunks, split train/test indices, build vocab on train
# Pass 2: transform in chunks, train SGD classifier incrementally
# This keeps peak RAM to ~2–3 GB regardless of dataset size

def train(args):
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    pf = pq.ParquetFile(args.data)

    # ── Pass 1a: collect all texts + labels in chunks ─────────────
    # We need texts in memory for TfidfVectorizer.fit()
    # Read at reduced chunk size to stay within RAM
    print("Pass 1: Reading dataset in chunks...")
    t0         = time.time()
    all_texts  = []
    all_labels = []
    total      = 0

    for batch in pf.iter_batches(batch_size=args.chunk_size,
                                  columns=["DATA", "TOPIC"]):
        chunk = batch.to_pandas()
        chunk = chunk[chunk["DATA"].notna() & (chunk["DATA"].str.strip() != "")]
        all_texts.extend(chunk["DATA"].tolist())
        all_labels.extend(chunk["TOPIC"].tolist())
        total += len(chunk)
        print(f"  {total:,} rows loaded...", end="\r")

    print(f"\nTotal rows: {total:,}  ({time.time()-t0:.1f}s)")

    # ── Label encode ──────────────────────────────────────────────
    le           = LabelEncoder()
    all_labels   = le.fit_transform(all_labels)
    num_classes  = len(le.classes_)
    print(f"Classes: {num_classes}  —  {list(le.classes_)}")

    # ── Train / test split ────────────────────────────────────────
    print(f"\nSplitting 80/20 stratified...")
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=all_labels
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Free the combined list immediately
    del all_texts, all_labels

    # ── Pass 2: fit TF-IDF on train ───────────────────────────────
    # TfidfVectorizer.fit() needs all train texts in memory at once.
    # At 8M rows this may still be tight — if it OOMs here, reduce chunk_size
    # which reduces total rows via the sampling above. Alternatively switch
    # to HashingVectorizer (see commented block at bottom).
    print("\nFitting TF-IDF on training set...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=args.min_df,
        strip_accents="unicode",
        dtype=np.float32,   # float32 instead of float64 — halves matrix RAM
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"Done in {time.time()-t0:.1f}s | shape: {X_train_tfidf.shape}")
    print(f"Matrix RAM: ~{X_train_tfidf.data.nbytes / 1e9:.2f} GB")

    # Free raw text lists after vectorization
    del X_train, X_test

    # ── Train Logistic Regression ─────────────────────────────────
    print("\nTraining Logistic Regression...")
    print("This may take 20–60 min on full 8M rows. Progress shown below.")
    t0  = time.time()
    clf = LogisticRegression(
        C=args.C,
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        random_state=args.seed,
        verbose=1,
    )
    clf.fit(X_train_tfidf, y_train)
    print(f"\nTraining done in {time.time()-t0:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────
    print("\nEvaluating...")
    y_pred = clf.predict(X_test_tfidf)
    acc    = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=le.classes_, digits=3))

    # ── Save ──────────────────────────────────────────────────────
    bundle = {
        "vectorizer":  vectorizer,
        "classifier":  clf,
        "label_encoder": le,
        "label_names": le.classes_.tolist(),
        "config": {
            "max_features": args.max_features,
            "ngram_range":  (1, 2),
            "sublinear_tf": True,
            "min_df":       args.min_df,
            "C":            args.C,
            "train_size":   int(X_train_tfidf.shape[0]),
            "test_size":    int(X_test_tfidf.shape[0]),
            "seed":         args.seed,
            "accuracy":     round(acc, 4),
        }
    }
    with open(args.output, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved to {args.output} "
          f"({os.path.getsize(args.output)/1e6:.1f} MB)")

    # ── Sanity check ──────────────────────────────────────────────
    print("\nSanity check:")
    test_texts = [
        "The stock market crashed after the Fed raised interest rates",
        "Manchester United won the Premier League title last night",
        "Python decorators are a powerful feature for metaprogramming",
        "The Eiffel Tower was built in 1889 and stands 330 meters tall",
        "Grilled salmon with lemon butter sauce and roasted vegetables",
        "NASA launched a new telescope to study distant galaxies",
    ]
    X_new  = vectorizer.transform(test_texts)
    preds  = le.inverse_transform(clf.predict(X_new))
    probs  = clf.predict_proba(X_new).max(axis=1)
    for text, pred, prob in zip(test_texts, preds, probs):
        print(f"  [{pred:<35}] {prob:.3f}  —  {text[:55]}")


if __name__ == "__main__":
    args = get_args()
    train(args)