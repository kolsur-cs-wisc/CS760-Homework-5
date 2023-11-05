import numpy as np

class PCA():
    def __init__(self, X, d, type=''):
        self.X = X
        self.D = X.shape[1]
        self.d = d
        self.type = type

    def demean(self):
        self.mean = np.mean(self.X, axis=0)
        demeaned_data = self.X - self.mean
        return demeaned_data
    
    def normalize(self):
        self.mean = np.mean(self.X, axis=0)
        self.standard_deviation = np.std(self.X, axis=0)
        self.standard_deviation[self.standard_deviation == 0] = 1.0
        normalized_data = (self.X - self.mean) / self.standard_deviation
        return normalized_data
    
    def transform(self):
        X_new = self.X
        if self.type == 'demean':
            X_new = self.demean()
        if self.type == 'normalize':
            X_new = self.normalize()
        
        U, S, VT = np.linalg.svd(X_new)
        V = np.reshape(VT[:self.d], (self.D, self.d))
        X_reduced = np.dot(X_new, V)
        self.X_reduced = X_reduced
        self.eigen_vectors = V

        X_recons = self.reconstruct()

        if self.type == 'demean':
            X_recons = X_recons + self.mean
        if self.type == 'normalize':
            X_recons = (X_recons * self.standard_deviation) + self.mean

        return X_reduced, X_recons
    
    def reconstruct(self):
        X_reconstructed = np.dot(self.X_reduced, self.eigen_vectors.transpose())
        return X_reconstructed
    
    def reconstruction_error(self):
        return np.sum((self.X - self.reconstruct())**2)/len(self.X)
