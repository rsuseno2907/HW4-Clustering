import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # print(y)
        score = []
        for i in range(len(X)):
            point = X[i]
            cluster = y[i]

            # Calculate a - how far is this point from other points that are in the same cluster
            a = None
            same_cluster_points = X[y == cluster]
            distances_in_cluster = np.linalg.norm(same_cluster_points - point, axis=1)
            if len(distances_in_cluster) > 1:
                a = np.sum(distances_in_cluster) / (len(distances_in_cluster) - 1)
            else:
                a = 0

            # Calculate b - how far is the smallest mean distance to a different cluster 
            b = None
            other_clusters = [c for c in np.unique(y) if c != cluster] # list other clusters that are not this point's
            if not other_clusters: # if there are no other clusters, set b to zero
                b = 0
            else: # otherwise, calculate the distance
                mean_distances_to_others = []
                for other_c in other_clusters:
                    other_cluster_points = X[y == other_c] #find points that are in the other cluster
                    dist_to_other = np.linalg.norm(other_cluster_points - point, axis=1) #distance from point of interest to points in that other cluster
                    mean_distances_to_others.append(np.mean(dist_to_other))
                b = min(mean_distances_to_others) #smalles mean distance

            s = (b-a)/(max(b,a))
            score.append(s)
            # print(s)
        # return np.mean(score)
        return score

                


