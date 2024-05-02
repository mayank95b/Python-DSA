"""Hopkins Statistic"""

# Authors:  Brian Hopkins
#           John Gordon Skellam

# URL Student : Manel Rodriguez Soto


import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets

from scipy.spatial import distance_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def visual_assessment(data):
    """Visual Assessment.
       Calculates the distance matrix of a data set and shows it as a heatmap."""

    dm = distance_matrix(data, data)
    sns.heatmap(dm, cmap="YlGnBu")

    plt.show()


def hopkins(x, n):
    """Hopkins Statistic
       Measures the cluster tendency of a data set.  Returns the statistic value.

       Parameters
       ----------
       x : array-like or sparse matrix, shape (n_samples, n_features). The data set

       n : int, number of points to sample randomly from the data set

    """

    d = len(x)
    columns = x.shape[1]

    for i in range(columns):
        if np.max(x[:, i]) == np.min(x[:, i]) == 0:
            pass
        else:
            x[:, i] = (x[:, i] - np.min(x[:, i])) / (np.max(x[:, i]) - np.min(x[:, i]))

    s = np.random.choice(d, n, replace=False)
    nns = NearestNeighbors(n_neighbors=1).fit(x)

    u, w = 0, 0
    for i in s:
        ran_point = [np.random.uniform(size=columns)]
        u += nns.kneighbors(ran_point)[0][0][0]
        w += nns.kneighbors([x[i]], 2)[0][0][1]

    return u / (u + w)


if __name__ == "__main__":
    # Example use

    data_set = datasets.load_iris()['data']

    sample_len = 10
    H = hopkins(data_set, sample_len)
    print(H)

    visual_assessment(data_set)
