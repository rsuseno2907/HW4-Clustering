import pytest
import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)

def test_consistent_seed():
    # If I run two instances of kmeans clustering with the same seed, it should give me same labels
    k = 4
    clusters, labels = make_clusters(k=k, scale=1)
    np.random.seed(42)
    km1 = KMeans(k=k)
    km1.fit(clusters)
    labels1 = km1.predict(clusters)

    np.random.seed(42)
    km2 = KMeans(k=k)
    km2.fit(clusters)
    labels2 = km2.predict(clusters)

    assert np.array_equal(labels1, labels2)

def test_num_cluster():
    # Test that number of unique labels are less than or equal to k
    k = 4
    clusters, labels = make_clusters(k=k, scale=1)
    km1 = KMeans(k=k)
    km1.fit(clusters)
    pred1 = km1.predict(clusters)

    assert len(np.unique(pred1) <= k) 

def test_centroid_validity():
    # Test that centroid is the correct shape
    k = 4
    clusters, labels = make_clusters(k=k, scale=1)
    km1 = KMeans(k=k)
    km1.fit(clusters)
    assert km1.centroids.shape == (k,2)

def test_single_cluster():
    # if only a single cluster, test that it return only label 0
    k = 1
    clusters, labels = make_clusters(k=k, scale=1)
    km1 = KMeans(k=k)
    km1.fit(clusters)
    pred1 = km1.predict(clusters)

    assert np.all(pred1 == np.int64(0))

