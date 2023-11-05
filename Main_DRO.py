import numpy as np
from DRO import DRO
from matplotlib import pyplot as plt

def plotData(X, X_recons):
    plt.scatter(X[:, 0], X[:, 1], marker = 'o', facecolors = 'none', edgecolors = 'blue',)
    plt.scatter(X_recons[:, 0], X_recons[:, 1], c = 'red', marker = 'x')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'{type} PCA')
    plt.savefig(f'output/DRO_data2D.png')
    plt.show()

def main():
    X = np.loadtxt('data/data2D.csv', delimiter=',')
    
    pca_model = DRO(X, 1)
    X_pca, X_recons = pca_model.transform()
    plotData(X, X_recons)

    pca_model = DRO(X, 1, 'demean')
    X_pca, X_recons = pca_model.transform()
    plotData(X, X_recons)

    pca_model = DRO(X, 1, 'normalize')
    X_pca, X_recons = pca_model.transform()
    plotData(X, X_recons)

if __name__ == '__main__':
    main()