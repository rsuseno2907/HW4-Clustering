import pytest
import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)
from sklearn.metrics import silhouette_score

def test_score_to_sklearn():
    k = 3
    clusters, labels = make_clusters(k=k, scale=1)
    km = KMeans(k=k)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    scores = np.mean(scores)
    sklearn_scores = silhouette_score(clusters, pred)
    assert np.isclose(scores, sklearn_scores, rtol=1e-4)
    # print(scores, sklearn_scores)