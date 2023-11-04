import numpy as np

def main():
    sigmas = [0.5, 1, 2, 4, 8]
    means = [[-1, -1], [1, -1], [0, 1]]
    covariances = [[[2, 0.5],[0.5, 1]],[[1, -0.5],[-0.5, 2]], [[1, 0],[0, 2]]]

    for sigma in sigmas:
        f = open(f'dataset_{sigma}.txt','a')
        for i in range(3):
            mean = np.array(means[i])
            covariance = np.array(covariances[i]) * sigma
            curr_sample = np.random.multivariate_normal(mean, covariance, 100)
            np.savetxt(f, curr_sample, delimiter = ' ')

    pass

if __name__ == "__main__":
    main()