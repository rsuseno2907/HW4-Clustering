import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int):
            raise TypeError("k must be an integer!")
        self.centroids = None
        self.k = k
        self.tol = tol
        self.assignment = None
        self.labels = None
        self.max_iter = max_iter

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        def mean_2d(points):
            return np.mean(points, axis=0)
        centroids = mat[np.random.choice(len(mat), self.k, replace=False)] # randomly choose k points as centroids
        label_fin = []
        for _ in range(self.max_iter): #safeguard so it doesn't loop endlessly
            labels = []

            # Assign each point to a centroid
            centroids_dict = {tuple(p): [] for p in centroids}
            for point in mat:
                distances = [self.dist(point, c) for c in centroids]
                idx = np.argmin(distances)
                labels.append(idx)
                centroids_dict[tuple(centroids[idx])].append(point)

            # Calculate new centroids
            new_centroids = []
            for c in centroids_dict:
                if len(centroids_dict[c]) == 0:
                    new_centroids.append(np.array(c))
                else:
                    new_centroids.append(mean_2d(centroids_dict[c]))

            # Check if new centroid is close to old centroid. if yes, break the loop
            shifts = np.linalg.norm(new_centroids - centroids, axis=1)
            if np.all(shifts < self.tol):
                break

            centroids = np.array(new_centroids)
        label_fin = labels
        self.centroids = centroids
        self.labels = label_fin
        print(centroids)


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        labels = []
        cens = []
        for point in mat:
            distances = [self.dist(point, c) for c in self.centroids]
            idx = np.argmin(distances)
            labels.append(idx)
            belong_to = self.centroids[idx]
            cens.append(belong_to)
            
        self.assignment = list(zip(mat,cens))
        return labels

    def get_error(self, mat) -> float: #TODO ask TA if it's okay to pass in `mat` here
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        distances_point_to_centroid = []
        for pair in self.assignment:
            distances_point_to_centroid.append(self.dist(pair[0], pair[1]))
            
        return np.mean(distances_point_to_centroid)



    def get_centroids(self) -> np.ndarray: # TODO check with TA if ok with output like this when i print the return
        # [[ 7.57041575 -2.9889736 ]
        #  [ 8.90376386  3.92500241]
        #  [ 6.51760379 -4.3566827 ]
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids

    def dist(self, p1, p2):
        return np.linalg.norm(p1-p2)


# clusters, labels = make_clusters(k=4, scale=1)
# km = KMeans(k=4)
# km.fit(clusters)
# pred = km.predict(clusters)
# km.get_error(clusters)
# print(km.get_centroids())
