from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import io

from engine import (
    build_user_embeddings,
    compute_lic_star,
    compute_momentum,
    compute_propagation
)

app = FastAPI(
    title="TransformIQ API",
    description="Transformation Intelligence Engine",
    version="1.0"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health():
    return {"status": "TransformIQ running"}


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    embeddings = build_user_embeddings(df)

    feature_cols = [
        col for col in embeddings.columns
        if col not in ["user_id", "time_window"]
    ]

    X = embeddings[feature_cols].fillna(0).values

    embeddings["lic_star"] = compute_lic_star(X)

    momentum_scores = compute_momentum(embeddings)

    propagation_scores = compute_propagation(
        X,
        momentum_scores,
        embeddings["user_id"].values
    )

    result = []

    for user in momentum_scores:

        result.append({
            "user_id": int(user),
            "momentum_score": momentum_scores[user],
            "propagation_score": propagation_scores.get(user, 0)
        })

    return {
        "n_users_analyzed": len(result),
        "results": sorted(
            result,
            key=lambda x: x["momentum_score"],
            reverse=True
        )
    }

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_transformation_map(X, lic_star, momentum):

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    sizes = np.array(momentum) * 500 + 20

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        s=sizes,
        c=lic_star
    )

    plt.colorbar(scatter, label="LIC*")
    plt.xlabel("Behavioral Dimension 1")
    plt.ylabel("Behavioral Dimension 2")
    plt.title("Transformation Tension Map")

    plt.show()

