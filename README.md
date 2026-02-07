[![BuildStatus](https://github.com/rsuseno2907/HW4-Clustering/workflows/HW4-Clustering/badge.svg?event=push)](https://github.com/rsuseno2907/HW4-Clustering/actions)

# K-Means Clustering
This repository is written for HW4 of the BMI203 Algorithms class assignment during Winter quarter of 2026.

Implemented Lloyd's algorithm for k-means clustering algorithm to cluster points based on a given `k`.

## What this project does
- Create a 2D cluster with `make_cluster()` function under `utils.py`
- Uses LLoyd's algorithm to find k centers under `KMeans.fit()` function
- Assign points to centers found under `KMeans.predict()` function
- Calculate Silhouette score to quantify "goodness" of clusters under `Silhouette.score()` function
- Run GitHub Actions (unit tests) automatically on every push

## Files
- `main.py`: wrapper script that runs everything
- `cluster/kmeans.py`: kmeans fit and predict implementation
- `cluster/silhouette.py`: silhouette score calculation
- `cluster/utils.py`: utility functions - create clusters and plot predicted clustering (as well as ground truth)

## Packages (Python 3.11)
- pytest
- numpy
- scipy
- sklearn