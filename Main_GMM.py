import numpy as np
from GMM import GMM
from matplotlib import pyplot as plt

def plotResultsClusteringAcc(sigmas, accuracies):
    plt.plot(sigmas, accuracies)
    plt.xlabel('Sigma')
    plt.ylabel('Clustering Accuracy')
    plt.savefig('output/clusterAccGMM.png')
    plt.show()

def plotResultsClusteringObj(sigmas, objectives):
    plt.plot(sigmas, objectives)
    plt.xlabel('Sigma')
    plt.ylabel('Clustering Objective')
    plt.savefig('output/clusterObjGMM.png')
    plt.show()

def main():
    sigmas = [0.5, 1, 2, 4, 8]
    results = []
    objectives = []
    accuracies = []
    for sigma in sigmas:
        X_train = np.loadtxt(f'data/dataset_{sigma}.txt', delimiter=' ', dtype=float)
        gmm_model = GMM(X_train)
        means, objective, accuracy = gmm_model.train()
        objectives.append(objective)
        accuracies.append(accuracy)
        results.append(f'Sigma = {sigma}, Cluster Centers = {means}, Clustering Objective = {objective}, Clustering Accuracy = {accuracy}')
    np.savetxt('Results GMM.txt', results, delimiter=' ', fmt='%s')
    plotResultsClusteringAcc(sigmas, accuracies)
    plotResultsClusteringObj(sigmas, objectives)

if __name__ == "__main__":
    main()