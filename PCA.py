import numpy as np

class PCA():
    def __init__(self, X, d, type=''):
        self.X = X
        self.d = d
        self.type = type

    def demean(self):
        mean = np.mean(self.X, axis=0)
        demeaned_data = self.X - mean
        return demeaned_data
    
    def normalize(self):
        mean = np.mean(self.X, axis=0)
        standard_deviation = np.std(self.X, axis=0)
        standard_deviation[standard_deviation == 0] = 1.0
        normalized_data = (self.X - mean) / standard_deviation
        return normalized_data
    
    def transform(self):
        X_new = self.X
        if self.type == 'demean':
            X_new = self.demean()
        if self.type == 'normalize':
            X_new = self.normalize()
        
        cov_mat = np.cov(X_new , rowvar = False)
        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

        sorted_index = np.argsort(eigen_values)[: : -1]
 
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        eigenvector_subset = sorted_eigenvectors[:, 0:self.d]

        X_reduced = np.dot(eigenvector_subset.transpose() , X_new.transpose()).transpose()

        self.X_reduced = X_reduced
        self.eigen_vectors = eigenvector_subset
        return X_reduced