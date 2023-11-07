import numpy as np

class DRO():
    def __init__(self, X, d) -> None:
        self.X = X
        self.D = X.shape[1]
        self.d = d

    def transform(self):
        self.b = np.mean(self.X, axis = 0)
        X_center = self.X - self.b
        U, S, VT = np.linalg.svd(X_center)
        self.A = np.reshape(VT[:self.d], (self.D, self.d))
        self.Z = np.dot(X_center, self.A)
        return self.Z, self.reconstruct(), S
    
    def reconstruct(self):
        self.X_recons = np.dot(self.Z, self.A.transpose()) + self.b
        return self.X_recons

    def reconstruction_error(self):
        return np.sum((self.X - self.reconstruct())**2)/len(self.X)
