import numpy as np

class DRO():
    def __init__(self, X, d) -> None:
        self.X = X
        self.D = X.shape[1]
        self.d = d

    def transform(self):
        n = len(self.X)

        self.b = np.mean(self.X, axis = 0)
        X_center = self.X - self.b
        U, S, VT = np.linalg.svd(X_center)
        SM = np.diag(S[:self.d])
        self.Z = np.sqrt(n) * U[:, :self.d]
        self.A = np.dot((np.sqrt(1/n) * SM), VT[:self.d]).T
        return self.Z, self.reconstruct(), S 
    
    def reconstruct(self):
        self.X_recons = np.dot(self.Z, self.A.transpose()) + self.b
        return self.X_recons

    def reconstruction_error(self):
        return np.sum((self.X - self.reconstruct())**2)/len(self.X)
