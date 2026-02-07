import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)


def main():

    # create tight clusters
    clusters, labels = make_clusters(scale=0.3)
    plot_clusters(clusters, labels, filename="figures/tight_clusters.png")

    # create loose clusters
    clusters, labels = make_clusters(scale=2)
    plot_clusters(clusters, labels, filename="figures/loose_clusters.png")

    """
    uncomment this section once you are ready to visualize your kmeans + silhouette implementation
    """
    k = 3
    clusters, labels = make_clusters(k=k, scale=1)
    km = KMeans(k=k)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    plot_multipanel(clusters, labels, pred, scores, filename='cluster.png')
    

if __name__ == "__main__":
    main()
