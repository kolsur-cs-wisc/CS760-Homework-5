import numpy as np
from KMeans import KMeans

def generate_datasets():
    sigmas = [0.5, 1, 2, 4, 8]
    labels = ['a', 'b', 'c']
    means = [[-1, -1], [1, -1], [0, 1]]
    covariances = [[[2, 0.5],[0.5, 1]],[[1, -0.5],[-0.5, 2]], [[1, 0],[0, 2]]]

    for sigma in sigmas:
        for i in range(3):
            mean = np.array(means[i])
            covariance = np.array(covariances[i]) * sigma
            curr_sample = np.random.multivariate_normal(mean, covariance, 100)
            f = open(f'dataset_{sigma}_labelled.txt', 'a')
            np.savetxt(f, curr_sample, delimiter = ' ')
            f.close()

def main():
    generate_datasets()
    X_train = np.loadtxt(f'dataset_0.5.txt', delimiter=' ', dtype=float)
    kmeans_model = KMeans()
    centroids, clusters = kmeans_model.fit(X_train)
    objective = kmeans_model.objective()
    print(f'Cluster Centers = {centroids}, Clustering Objective = {objective}')

if __name__ == "__main__":
    main()