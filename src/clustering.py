
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def build_wine_pipeline(n_clusters=3, variance=0.95):
    """
    Build a complete Wine clustering pipeline.

    Steps:
    1. Standardise the input features.
    2. Apply PCA to retain the specified variance.
    3. Perform K-Means clustering.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=variance)),
        ("kmeans", KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        ))
    ])

    return pipeline
