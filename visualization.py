import matplotlib
matplotlib.use("Agg")  # Backend non-GUI pour serveur
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import io
import base64

def generate_transformation_map(X, lic_star, momentum_scores_dict, user_ids):
    """
    X : array (n_users, n_features)
    lic_star : array (n_users,)
    momentum_scores_dict : dict {user_id: score}
    user_ids : array-like correspondant à l'ordre de X

    Retourne : PNG encodé en base64
    """

    # 1️⃣ PCA 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 2️⃣ Alignement des scores avec l'ordre de X
    momentum_array = np.array([momentum_scores_dict[user] for user in user_ids])
    sizes = momentum_array * 500 + 20  # taille des points

    # 3️⃣ Scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        s=sizes,
        c=lic_star,
        cmap="viridis",
        alpha=0.7
    )
    plt.colorbar(scatter, label="LIC*")
    plt.title("Transformation Tension Map")
    plt.xlabel("Comportement Dimension 1")
    plt.ylabel("Comportement Dimension 2")

    # 4️⃣ Export en base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return image_base64
