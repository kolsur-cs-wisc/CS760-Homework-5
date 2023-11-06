import numpy as np
from DRO import DRO
from matplotlib import pyplot as plt

def plotData(X, X_recons):
    plt.scatter(X[:, 0], X[:, 1], marker = 'o', facecolors = 'none', edgecolors = 'blue',)
    plt.scatter(X_recons[:, 0], X_recons[:, 1], c = 'red', marker = 'x')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('DRO Method')
    plt.savefig(f'output/DRO_data2D.png')
    plt.show()

def main():
    X = np.loadtxt('data/data2D.csv', delimiter=',')

    dro_model = DRO(X, 1)
    X_pca, X_recons = dro_model.transform()
    plotData(X, X_recons)
    np.savetxt('output/DRO Results.txt', [f'DRO 2D Dataset Reconstruction Error = {dro_model.reconstruction_error()}'], fmt = '%s')

if __name__ == '__main__':
    main()