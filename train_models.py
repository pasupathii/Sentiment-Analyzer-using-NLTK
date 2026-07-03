"""
train_models.py – Multi-model training pipeline with comparison and auto-selection.

Models trained:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM (LinearSVC with CalibratedClassifierCV for probabilities)
4. Random Forest
5. Voting Ensemble (top 3 models)

Saves best model + vectorizer + training report to models/ directory.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline
import joblib

from data_loader import load_and_prepare, LABEL_NAMES

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# TF-IDF settings
TFIDF_PARAMS = {
    "max_features": 25000,
    "ngram_range": (1, 3),
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
}


# ---------------------------------------------------------------------------
# Model definitions with hyperparameter grids
# ---------------------------------------------------------------------------

def get_models():
    """Return dict of model_name → (estimator, param_grid)."""
    return {
        "Naive Bayes": (
            MultinomialNB(),
            {"alpha": [0.01, 0.1, 0.5, 1.0]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs",
                               class_weight="balanced"),
            {"C": [1.0, 5.0, 10.0, 50.0]},
        ),
        "Linear SVM": (
            CalibratedClassifierCV(
                LinearSVC(max_iter=3000, random_state=RANDOM_STATE, class_weight="balanced"),
                cv=3,
            ),
            # CalibratedClassifierCV wraps LinearSVC; params go through estimator__
            {"estimator__C": [0.5, 1.0, 5.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced",
                                   n_jobs=-1),
            {
                "n_estimators": [100],
                "max_depth": [None],
                "min_samples_split": [2],
            },
        ),
    }


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train_and_evaluate(verbose: bool = True):
    """
    Full training pipeline.

    Returns
    -------
    best_model, vectorizer, report : tuple
    """
    # ── Load data ──────────────────────────────────────────────────────────
    texts, labels = load_and_prepare(balance=True, verbose=verbose)

    # ── Vectorize ──────────────────────────────────────────────────────────
    if verbose:
        print("\n[*] Vectorizing with TF-IDF ...")
        print(f"   Params: {TFIDF_PARAMS}")

    from nltk.corpus import stopwords
    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words, **TFIDF_PARAMS)

    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    if verbose:
        print(f"   Feature matrix: {X.shape}")

    # ── Train / Test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    if verbose:
        print(f"\n[*] Split: {X_train.shape[0]} train / {X_test.shape[0]} test")

    # ── Train each model ───────────────────────────────────────────────────
    models = get_models()
    results = {}
    trained_models = {}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for name, (estimator, param_grid) in models.items():
        if verbose:
            print(f"\n[>] Training: {name} ...")

        t0 = time.time()

        grid = GridSearchCV(
            estimator, param_grid, cv=cv, scoring="f1_weighted",
            n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        train_time = time.time() - t0

        # Predict
        y_pred = best.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "best_params": {str(k): str(v) for k, v in grid.best_params_.items()},
            "train_time_seconds": round(train_time, 2),
            "confusion_matrix": cm,
        }
        trained_models[name] = best

        if verbose:
            print(f"   Accuracy : {acc:.4f}")
            print(f"   F1 Score : {f1:.4f}")
            print(f"   Time     : {train_time:.1f}s")
            print(f"   Params   : {grid.best_params_}")

    # ── Build Voting Ensemble ──────────────────────────────────────────────
    if verbose:
        print("\n[>] Building Voting Ensemble (top 3) ...")

    # Pick top 3 by F1
    sorted_models = sorted(results.items(), key=lambda x: x[1]["f1_score"], reverse=True)
    top3_names = [name for name, _ in sorted_models[:3]]
    estimators_for_voting = [(name, trained_models[name]) for name in top3_names]

    if verbose:
        print(f"   Using: {top3_names}")

    t0 = time.time()
    voting = VotingClassifier(
        estimators=estimators_for_voting,
        voting="soft",
    )
    voting.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = voting.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    results["Voting Ensemble"] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "best_params": {"estimators": top3_names},
        "train_time_seconds": round(train_time, 2),
        "confusion_matrix": cm,
    }
    trained_models["Voting Ensemble"] = voting

    if verbose:
        print(f"   Accuracy : {acc:.4f}")
        print(f"   F1 Score : {f1:.4f}")

    # ── Select best overall model ──────────────────────────────────────────
    best_name = max(results, key=lambda x: results[x]["f1_score"])
    best_model = trained_models[best_name]

    if verbose:
        print(f"\n[BEST] Best model: {best_name} (F1={results[best_name]['f1_score']:.4f})")
        print("\n[*] Full classification report (best model):")
        y_pred_best = best_model.predict(X_test)
        target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
        print(classification_report(y_test, y_pred_best, target_names=target_names))

    # ── Save artifacts ─────────────────────────────────────────────────────
    if verbose:
        print("[*] Saving models and report ...")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"), compress=3)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"), compress=3)

    report = {
        "best_model_name": best_name,
        "best_f1_score": results[best_name]["f1_score"],
        "best_accuracy": results[best_name]["accuracy"],
        "total_training_samples": int(X_train.shape[0]),
        "total_test_samples": int(X_test.shape[0]),
        "tfidf_features": int(X.shape[1]),
        "label_names": LABEL_NAMES,
        "trained_at": datetime.now().isoformat(),
        "all_results": results,
    }
    with open(os.path.join(MODELS_DIR, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"   [OK] Saved to {MODELS_DIR}/")
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Best Model  : {best_name}")
        print(f"  Accuracy    : {results[best_name]['accuracy']:.2%}")
        print(f"  F1 Score    : {results[best_name]['f1_score']:.2%}")
        print("=" * 60)

    return best_model, vectorizer, report


# ---------------------------------------------------------------------------
# Quick-test function
# ---------------------------------------------------------------------------


def quick_test(model, vectorizer):
    """Test the model with sample sentences."""
    from data_loader import clean_text

    test_sentences = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible experience. The service was awful and rude.",
        "The package arrived on time. It was okay, nothing special.",
        "I love this app, it makes my life so much easier!",
        "Worst customer service I have ever experienced. Never again.",
        "The food was decent. Average quality for the price.",
    ]

    print("\n[*] Quick Test Results:")
    print("-" * 60)
    for sentence in test_sentences:
        cleaned = clean_text(sentence)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        label = LABEL_NAMES[pred]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            conf = max(proba) * 100
            print(f"  [{label:>8}] ({conf:5.1f}%) | {sentence[:55]}")
        else:
            print(f"  [{label:>8}]          | {sentence[:55]}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    best_model, vectorizer, report = train_and_evaluate(verbose=True)
    quick_test(best_model, vectorizer)
