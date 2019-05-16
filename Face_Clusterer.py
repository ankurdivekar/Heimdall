import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_N_centroids(encodings, n_clusters=10, show_plot=False):

    enc_array = np.squeeze(np.array(encodings))
    km_model = KMeans(n_clusters=n_clusters)
    km_model.fit(enc_array)

    centers = km_model.cluster_centers_

    y_kmeans = km_model.predict(enc_array)
    if show_plot:
        plt.scatter(enc_array[:, 15], enc_array[:, 1], c=y_kmeans, s=5, cmap='viridis')
        plt.scatter(centers[:, 15], centers[:, 1], c='black', s=50, alpha=0.8)
        plt.show()

    return centers
