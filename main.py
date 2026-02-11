from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io

from engine import (
    build_user_embeddings,
    compute_lic_star,
    compute_momentum,
    compute_propagation
)
from visualization import generate_transformation_map

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
    """
    Vérifie que l'API fonctionne
    """
    return {"status": "TransformIQ running"}


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

from fastapi.responses import StreamingResponse
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    embeddings = build_user_embeddings(df)
    feature_cols = [col for col in embeddings.columns if col not in ["user_id", "time_window"]]
    X = embeddings[feature_cols].fillna(0).values
    embeddings["lic_star"] = compute_lic_star(X)
    momentum_scores = compute_momentum(embeddings)

    # Alignement
    user_ids = embeddings["user_id"].values
    momentum_array = np.array([momentum_scores[user] for user in user_ids])
    lic_star_array = embeddings["lic_star"].values

    # PCA 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(X_2d[:,0], X_2d[:,1], s=momentum_array*500+20, c=lic_star_array, cmap="viridis", alpha=0.7)
    plt.colorbar(label="LIC*")
    plt.title("Transformation Tension Map")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # Retour en image/png
    return StreamingResponse(buf, media_type="image/png")
