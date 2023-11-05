import numpy as np
from GMM import GMM

def main():
    X_train = np.loadtxt(f'data/dataset_1.txt', delimiter=' ', dtype=float)
    gmm_model = GMM()
    means, ll = gmm_model.train(X_train)
    print(means, ll)

if __name__ == "__main__":
    main()