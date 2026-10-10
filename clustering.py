"""Reusable K-means fitting with the experiment's original seeds and initialization."""

import numpy as np
from sklearn.cluster import KMeans


def validate_features(features):
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite income/spending features with shape (N, 2)")
    return values


def elbow_inertias(features, max_clusters=10):
    values = validate_features(features)
    if not 1 <= max_clusters <= len(values):
        raise ValueError("Cluster counts must be between one and the sample count")
    return [
        KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0).fit(values).inertia_
        for k in range(1, max_clusters + 1)
    ]


def segment_customers(features, clusters=5):
    values = validate_features(features)
    if not 1 <= clusters <= len(values):
        raise ValueError("Cluster count must be between one and the sample count")
    model = KMeans(n_clusters=clusters, init="k-means++", n_init=10, random_state=42)
    return model, model.fit_predict(values)
