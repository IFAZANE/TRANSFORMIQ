import matplotlib.pyplot as plt
matplotlib.use("Agg")  # Backend non-GUI pour serveur
import numpy as np
from sklearn.decomposition import PCA
import base64
import io


def generate_transformation_map(X, lic_star, momentum):

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    sizes = np.array(momentum) * 500 + 20

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        s=sizes,
        c=lic_star
    )

    plt.colorbar(scatter, label="LIC*")
    plt.title("Transformation Tension Map")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return image_base64

