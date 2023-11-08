import numpy as np
from scipy import rand
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, X, k=3, dim=2):
        self.k = k
        self.mu = rand(k, dim)*1 - 0.5
        init_sigma = np.zeros((k, dim, dim))
        for i in range(k): init_sigma[i] = np.eye(dim)
        self.sigma = init_sigma
        self.pi = np.ones(self.k)/self.k
        self.X = X
        self.n = len(X)
        self.z = np.zeros((self.n, self.k))
    
    def e_step(self):
        for i in range(self.k):
            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.X, mean=self.mu[i], cov=self.sigma[i])
        self.z /= self.z.sum(axis=1, keepdims=True)
    
    def m_step(self):
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.n
        self.mu = np.matmul(self.z.T, self.X)
        self.mu /= sum_z[:, None]
        for i in range(self.k):
            j = np.expand_dims(self.X, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i])
            self.sigma[i] /= sum_z[i]
            
    def log_likelihood(self, X):
        ll = []
        for d in X:
            tot = 0
            for i in range(self.k):
                tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
            ll.append(np.log(tot))
        return np.sum(ll)
    
    def train(self):
        num_iters = 60
        log_likelihood = [self.log_likelihood(self.X)]
        for e in range(num_iters):
            self.e_step()
            self.m_step()
            log_likelihood.append(self.log_likelihood(self.X))

        return self.mu, -self.log_likelihood(self.X), self.accuracy()
    
    def euclidean(self, point, centroids):
        return np.sqrt(np.sum((point - centroids)**2, axis=1))
    
    def accuracy(self):
        distribution_means = [[-1, -1], [1, -1], [0, 1]]
        labels = ['a', 'b', 'c']
        label_dict = {}

        for i in range(len(self.mu)):
            center = self.mu[i]
            dist = self.euclidean(center, distribution_means)
            label_idx = np.argmin(dist)
            label_dict[i] = labels[label_idx]

        correct = 0
        
        for i in range(len(self.X)):
            x = self.X[i]
            label = 'a'
            if i >= 100: label = 'b'
            if i >= 200: label = 'c'

            dist = self.euclidean(x, self.mu)
            label_idx = np.argmin(dist)
            prediction = label_dict[label_idx]

            if label == prediction : correct += 1

        return correct/len(self.X)