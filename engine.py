import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


# ============================================================
# BUILD USER EMBEDDINGS
# ============================================================

def build_user_embeddings(df, time_window="7D"):

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    embeddings = []

    for window_start in pd.date_range(
        df.timestamp.min(),
        df.timestamp.max(),
        freq=time_window
    ):

        window_end = window_start + pd.Timedelta(time_window)

        window_df = df[
            (df.timestamp >= window_start) &
            (df.timestamp < window_end)
        ]

        if window_df.empty:
            continue

        grouped = (
            window_df
            .groupby(["user_id", "event_type"])
            .size()
            .unstack(fill_value=0)
        )

        grouped["total_events"] = grouped.sum(axis=1)
        grouped["event_diversity"] = (grouped > 0).sum(axis=1)
        grouped["time_window"] = window_start

        embeddings.append(grouped.reset_index())

    return pd.concat(embeddings, ignore_index=True)


# ============================================================
# LIC*
# ============================================================

def compute_lic_star(X, k=10):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nbrs = NearestNeighbors(n_neighbors=min(k, len(X_scaled)))
    nbrs.fit(X_scaled)

    distances, indices = nbrs.kneighbors(X_scaled)

    rarity = distances.mean(axis=1)

    instability = []
    for idx_list in indices:
        instability.append(np.var(X_scaled[idx_list], axis=0).mean())

    instability = np.array(instability)

    rarity_norm = rarity / (rarity.max() + 1e-8)
    instability_norm = instability / (instability.max() + 1e-8)

    return rarity_norm * instability_norm


# ============================================================
# MOMENTUM SCORE
# ============================================================

def compute_momentum(df_embeddings):

    df_embeddings = df_embeddings.sort_values(["user_id", "time_window"])

    momentum_scores = {}

    for user, group in df_embeddings.groupby("user_id"):

        lic_values = group["lic_star"].values
        diffs = np.diff(lic_values)
        positive_diffs = diffs[diffs > 0]

        momentum_scores[user] = float(positive_diffs.sum())

    return momentum_scores


# ============================================================
# PROPAGATION SCORE
# ============================================================

def compute_propagation(X, momentum_scores, user_ids, k=5):

    nbrs = NearestNeighbors(n_neighbors=min(k, len(X)))
    nbrs.fit(X)

    _, indices = nbrs.kneighbors(X)

    propagation_scores = {}

    for idx, neighbors in enumerate(indices):

        user = user_ids[idx]

        influence = 0
        for n_idx in neighbors:
            neighbor_user = user_ids[n_idx]
            influence += momentum_scores.get(neighbor_user, 0)

        propagation_scores[user] = float(influence)

    return propagation_scores
