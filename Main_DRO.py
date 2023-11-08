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

def plotSingularValues(S):
    plt.plot(S)
    plt.title("Singular Values for 1000D")
    plt.savefig('output/DRO Singular Values 1000D.png')
    plt.show()

def main():
    X = np.loadtxt('data/data2D.csv', delimiter=',')
    X1000 = np.loadtxt('data/data1000D.csv', delimiter=',')

    d = 30
    dro_model = DRO(X, 1)
    dro_model = DRO(X1000, d)
    X_pca, X_recons, Singular_Values = dro_model.transform()
    # plotData(X, X_recons)
    print(dro_model.reconstruction_error())
    plotSingularValues(Singular_Values)
    errors = []
    for d in range(0, 1001):
        dro_model = DRO(X1000, d)
        X_pca, X_recons, Singular_Values = dro_model.transform()
        
        errors.append(f'DRO 1000D Dataset Reconstruction Error with d {d} = {dro_model.reconstruction_error()}')
    np.savetxt('output/DRO 1000D Results.txt', errors, fmt = '%s')

if __name__ == '__main__':
    main()