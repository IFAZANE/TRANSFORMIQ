from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import io
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import plotly.express as px

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
# ANALYZE JSON
# ============================================================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Retourne JSON avec :
    - momentum_score
    - propagation_score
    - LIC*
    """

    # Lecture CSV
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Vérification colonnes essentielles
    required_cols = ["user_id", "timestamp", "event_type"]
    if not all(col in df.columns for col in required_cols):
        return {"error": f"Le fichier doit contenir : {required_cols}"}

    # Embeddings et scores
    embeddings = build_user_embeddings(df)
    feature_cols = [col for col in embeddings.columns if col not in ["user_id", "time_window"]]
    X = embeddings[feature_cols].fillna(0).values

    embeddings["lic_star"] = compute_lic_star(X)
    momentum_scores = compute_momentum(embeddings)
    propagation_scores = compute_propagation(X, momentum_scores, embeddings["user_id"].values)

    # Préparer JSON
    scores = [
        {
            "user_id": int(uid),
            "momentum_score": float(momentum_scores[uid]),
            "propagation_score": float(propagation_scores[uid]),
            "lic_star": float(embeddings.loc[embeddings['user_id']==uid, 'lic_star'].values[0])
        }
        for uid in embeddings["user_id"].values
    ]

    return JSONResponse({
        "n_users_analyzed": len(embeddings),
        "scores": scores
    })


# ============================================================
# ANALYZE IMAGE PNG
# ============================================================

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Retourne PNG statique de la carte (heatmap)
    """

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    embeddings = build_user_embeddings(df)
    feature_cols = [col for col in embeddings.columns if col not in ["user_id", "time_window"]]
    X = embeddings[feature_cols].fillna(0).values

    embeddings["lic_star"] = compute_lic_star(X)
    momentum_scores = compute_momentum(embeddings)

    user_ids = embeddings["user_id"].values
    momentum_array = np.array([momentum_scores[user] for user in user_ids])
    lic_star_array = embeddings["lic_star"].values

    # PCA 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Plot matplotlib
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

    return StreamingResponse(buf, media_type="image/png")


# ============================================================
# ANALYZE INTERACTIVE (Plotly)
# ============================================================

@app.post("/analyze_interactive")
async def analyze_interactive(file: UploadFile = File(...)):
    """
    Retourne JSON des scores + carte interactive Plotly en base64 HTML
    """

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    embeddings = build_user_embeddings(df)
    feature_cols = [col for col in embeddings.columns if col not in ["user_id", "time_window"]]
    X = embeddings[feature_cols].fillna(0).values

    embeddings["lic_star"] = compute_lic_star(X)
    momentum_scores = compute_momentum(embeddings)
    propagation_scores = compute_propagation(X, momentum_scores, embeddings["user_id"].values)

    user_ids = embeddings["user_id"].values
    momentum_array = np.array([momentum_scores[u] for u in user_ids])
    propagation_array = np.array([propagation_scores[u] for u in user_ids])
    lic_star_array = embeddings["lic_star"].values

    # PCA 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    df_plot = pd.DataFrame({
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "user_id": user_ids,
        "momentum": momentum_array,
        "propagation": propagation_array,
        "lic_star": lic_star_array
    })

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="lic_star",
        size="momentum",
        hover_data=["user_id", "momentum", "propagation", "lic_star"],
        color_continuous_scale="Viridis",
        size_max=25,
        title="Transformation Tension Map (Interactive)"
    )

    html_bytes = fig.to_html(full_html=False).encode("utf-8")
    html_base64 = base64.b64encode(html_bytes).decode("utf-8")

    scores = [
        {
            "user_id": int(uid),
            "momentum_score": float(momentum_scores[uid]),
            "propagation_score": float(propagation_scores[uid]),
            "lic_star": float(embeddings.loc[embeddings['user_id']==uid, 'lic_star'].values[0])
        }
        for uid in user_ids
    ]

    return JSONResponse({
        "n_users_analyzed": len(user_ids),
        "scores": scores,
        "interactive_map_base64": html_base64
    })
