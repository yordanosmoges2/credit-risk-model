from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Config
# ----------------------------
@dataclass
class RFMConfig:
    raw_path: str = "data/raw/data.csv"
    output_path: str = "data/processed/rfm_target.csv"
    customer_id_col: str = "CustomerId"
    datetime_col: str = "TransactionStartTime"
    amount_col: str = "Amount"
    n_clusters: int = 3


# ----------------------------
# Helpers
# ----------------------------
def load_raw_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at: {path}")
    return pd.read_csv(path)


def compute_rfm(df: pd.DataFrame, cfg: RFMConfig) -> pd.DataFrame:
    df = df.copy()
    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")

    snapshot_date = df[cfg.datetime_col].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby(cfg.customer_id_col)
        .agg(
            Recency=(cfg.datetime_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=(cfg.customer_id_col, "count"),
            Monetary=(cfg.amount_col, "sum"),
        )
        .reset_index()
    )

    return rfm


def cluster_rfm(rfm: pd.DataFrame, cfg: RFMConfig) -> pd.DataFrame:
    features = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    return rfm


def assign_risk_label(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Define high-risk customers as the cluster with:
    - Highest Recency (least recent)
    - Lowest Frequency
    - Lowest Monetary
    """
    cluster_stats = (
        rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
    )

    high_risk_cluster = cluster_stats.sort_values(
        by=["Recency", "Frequency", "Monetary"],
        ascending=[False, True, True],
    ).iloc[0]["cluster"]

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
    return rfm


def save_target(rfm: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rfm.to_csv(output_path, index=False)
    print(f"✅ Saved RFM target to: {output_path}")


# ----------------------------
# Main
# ----------------------------
def run_rfm(cfg: RFMConfig) -> None:
    df = load_raw_data(cfg.raw_path)
    rfm = compute_rfm(df, cfg)
    rfm = cluster_rfm(rfm, cfg)
    rfm = assign_risk_label(rfm)
    save_target(rfm, cfg.output_path)

    print("📊 Cluster distribution:")
    print(rfm["cluster"].value_counts())
    print("🎯 High-risk ratio:")
    print(rfm["is_high_risk"].mean())


if __name__ == "__main__":
    config = RFMConfig()
    run_rfm(config)
