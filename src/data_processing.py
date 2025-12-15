from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------
# Config
# ----------------------------
@dataclass
class ProcessingConfig:
    raw_path: str = "data/raw/data.csv"
    output_path: str = "data/processed/processed.csv"
    datetime_col: str = "TransactionStartTime"
    customer_id_col: str = "CustomerId"
    amount_col: str = "Amount"


# ----------------------------
# Custom Transformers
# ----------------------------
class CustomerAggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Creates customer-level aggregates and merges them back to each transaction row.
    Required aggregates (per customer):
    - total_amount, avg_amount, txn_count, std_amount
    """

    def __init__(self, customer_id_col: str, amount_col: str):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self._agg_df: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.customer_id_col not in X.columns:
            raise ValueError(f"Missing column: {self.customer_id_col}")
        if self.amount_col not in X.columns:
            raise ValueError(f"Missing column: {self.amount_col}")

        grp = X.groupby(self.customer_id_col)[self.amount_col]
        self._agg_df = pd.DataFrame({
            "total_amount": grp.sum(),
            "avg_amount": grp.mean(),
            "txn_count": grp.count(),
            "std_amount": grp.std(ddof=0).fillna(0.0),
        }).reset_index()
        return self

    def transform(self, X: pd.DataFrame):
        if self._agg_df is None:
            raise RuntimeError("Transformer not fitted.")
        return X.merge(self._agg_df, on=self.customer_id_col, how="left")


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts hour/day/month/year from a datetime column.
    Creates: txn_hour, txn_day, txn_month, txn_year
    """

    def __init__(self, datetime_col: str):
        self.datetime_col = datetime_col

    def fit(self, X: pd.DataFrame, y=None):
        if self.datetime_col not in X.columns:
            raise ValueError(f"Missing column: {self.datetime_col}")
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        dt = pd.to_datetime(X[self.datetime_col], errors="coerce", utc=True)

        X["txn_hour"] = dt.dt.hour
        X["txn_day"] = dt.dt.day
        X["txn_month"] = dt.dt.month
        X["txn_year"] = dt.dt.year

        return X


# ----------------------------
# Helpers
# ----------------------------
def load_raw_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at: {path}")
    return pd.read_csv(path)


def split_columns(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[List[str], List[str]]:
    # Numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target if present
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)
    if target_col and target_col in cat_cols:
        cat_cols.remove(target_col)

    return num_cols, cat_cols


def build_processing_pipeline(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor


def make_model_ready_dataset(df: pd.DataFrame, cfg: ProcessingConfig) -> Tuple[np.ndarray, List[str]]:
    # Feature creation at dataframe level
    df = CustomerAggregateFeatures(cfg.customer_id_col, cfg.amount_col).fit_transform(df)
    df = DateTimeFeatures(cfg.datetime_col).fit_transform(df)

    # Drop high-cardinality IDs that usually don’t help (optional but recommended)
    drop_cols = ["TransactionId", "BatchId", "AccountId", "SubscriptionId"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)

    # Choose columns for preprocessing
    num_cols, cat_cols = split_columns(df, target_col=None)

    preprocessor = build_processing_pipeline(num_cols, cat_cols)

    X = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return X, feature_names


from scipy import sparse


def save_processed(X, feature_names: List[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if sparse.issparse(X):
        # Save sparse matrix efficiently
        sparse.save_npz(output_path.replace(".csv", ".npz"), X)
        # Save feature names separately
        pd.Series(feature_names).to_csv(
            output_path.replace(".csv", "_features.csv"),
            index=False
        )
        print("ℹ️ Saved sparse matrix (.npz) and feature names.")
    else:
        out_df = pd.DataFrame(X, columns=feature_names)
        out_df.to_csv(output_path, index=False)



# ----------------------------
# Main entry
# ----------------------------
def run_processing(cfg: ProcessingConfig) -> None:
    df = load_raw_data(cfg.raw_path)
    X, feature_names = make_model_ready_dataset(df, cfg)
    save_processed(X, feature_names, cfg.output_path)
    print(f"✅ Saved processed dataset to: {cfg.output_path}")
    print(f"✅ Shape: {X.shape}")


if __name__ == "__main__":
    config = ProcessingConfig()
    run_processing(config)
