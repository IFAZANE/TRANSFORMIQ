from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io

from engine import (
    build_user_embeddings,
    compute_lic_star,
    compute_momentum,
    compute_propagation
)
from visualization import generate_transformation_map  # <- Import de la fonction de plot

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
    Endpoint de vérification que l'API fonctionne
    """
    return {"status": "TransformIQ running"}


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Endpoint principal pour analyser un CSV d'événements utilisateurs
    Retourne :
        - momentum_score
        - propagation_score
        - transformation_map_image (PNG encodée en base64)
    """

    # 1️⃣ Lecture du CSV
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Vérification des colonnes nécessaires
    required_cols = ["user_id", "timestamp", "event_type"]
    if not all(col in df.columns for col in required_cols):
        return {"error": f"Le fichier doit contenir les colonnes : {required_cols}"}

    # 2️⃣ Construction des embeddings utilisateurs
    embeddings = build_user_embeddings(df)

    feature_cols = [
        col for col in embeddings.columns
        if col not in ["user_id", "time_window"]
    ]

    X = embeddings[feature_cols].fillna(0).values

    # 3️⃣ Calcul LIC* pour chaque utilisateur
    embeddings["lic_star"] = compute_lic_star(X)

    # 4️⃣ Calcul du momentum score
    momentum_scores = compute_momentum(embeddings)

    # 5️⃣ Calcul du propagation score
    propagation_scores = compute_propagation(
        X,
        momentum_scores,
        embeddings["user_id"].values
    )

    # 6️⃣ Préparation des résultats pour retour API
    result = []
    for user in momentum_scores:
        result.append({
            "user_id": int(user),
            "momentum_score": momentum_scores[user],
            "propagation_score": propagation_scores.get(user, 0)
        })

    # 7️⃣ Génération du graphique de transformation (base64 PNG)
    image_base64 = generate_transformation_map(
        X,
        embeddings["lic_star"].values,
        list(momentum_scores.values())
    )

    # 8️⃣ Retour JSON
    return {
        "n_users_analyzed": len(result),
        "results": sorted(
            result,
            key=lambda x: x["momentum_score"],
            reverse=True
        ),
        "transformation_map_image": image_base64
    }
