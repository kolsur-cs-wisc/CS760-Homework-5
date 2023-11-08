import numpy as np
from matplotlib import pyplot as plt
from KMeans import KMeans

sigmas = [0.5, 1, 2, 4, 8]
means = [[-1, -1], [1, -1], [0, 1]]
covariances = [[[2, 0.5],[0.5, 1]],[[1, -0.5],[-0.5, 2]], [[1, 0],[0, 2]]]

def generate_datasets():
    for sigma in sigmas:
        for i in range(3):
            mean = np.array(means[i])
            covariance = np.array(covariances[i]) * sigma
            curr_sample = np.random.multivariate_normal(mean, covariance, 100)
            f = open(f'data/dataset_{sigma}_labelled.txt', 'a')
            np.savetxt(f, curr_sample, delimiter = ' ')
            f.close()

def plotResultsClusteringAcc():
    x = [0.5, 1, 2, 4, 8]
    y = [0.82, 0.68, 0.56, 0.5766666666666667, 0.4266666666666667]

    plt.plot(x, y)
    plt.xlabel('Sigma')
    plt.ylabel('Clustering Accuracy')
    plt.savefig('output/clusterAccKMeans.png')
    plt.show()

def plotResultsClusteringObj():
    x = [0.5, 1, 2, 4, 8]
    y = [326.14828206716464, 534.630041141039, 1069.8016111324475, 1470.8925872982986, 3584.8766593149785]

    plt.plot(x, y)
    plt.xlabel('Sigma')
    plt.ylabel('Clustering Objective')
    plt.savefig('output/clusterObjKMeans.png')
    plt.show()

def main():
    # generate_datasets()
    plotResultsClusteringAcc()
    plotResultsClusteringObj()
    results = []
    for sigma in sigmas:
        X_train = np.loadtxt(f'data/dataset_{sigma}.txt', delimiter=' ', dtype=float)
        kmeans_model = KMeans()
        centroids, clusters = kmeans_model.fit(X_train)
        objective, accuracy = kmeans_model.objective(), kmeans_model.accuracy()
        results.append(f'Sigma = {sigma}, Cluster Centers = {centroids}, Clustering Objective = {objective}, Clustering Accuracy = {accuracy}')
    np.savetxt('Results K Means.txt', results, delimiter=' ', fmt='%s')

if __name__ == "__main__":
    main()