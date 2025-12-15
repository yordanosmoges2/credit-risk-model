from __future__ import annotations

import os
from typing import Dict

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Config
# ----------------------------
DATA_PATH = "data/processed/rfm_target.csv"
TARGET_COL = "is_high_risk"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_DIR = "models"


# ----------------------------
# Helpers
# ----------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found at: {path}")
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    X = df[["Recency", "Frequency", "Monetary"]]
    y = df[TARGET_COL]
    return X, y


def evaluate(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# ----------------------------
# Training
# ----------------------------
def train_and_log(model, model_name: str, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"✅ {model_name} metrics:", metrics)


def main():
    mlflow.set_experiment("credit-risk-rfm")

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ----------------------------
    # Logistic Regression
    # ----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    train_and_log(
        log_reg,
        "LogisticRegression",
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )

    # ✅ SAVE MODEL + SCALER FOR API
    joblib.dump(log_reg, f"{MODEL_DIR}/logistic_regression.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    # ----------------------------
    # Random Forest
    # ----------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    train_and_log(
        rf,
        "RandomForest",
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()
