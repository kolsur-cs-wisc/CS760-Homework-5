import numpy as np

class KMeans():
    def __init__(self, k = 3):
        self.k = k
        self.max_iter = 500

    def euclidean(self, point, centroids):
        return np.sqrt(np.sum((point - centroids)**2, axis=1))
    
    def fit(self, X_train):
        self.X = X_train
        index = np.random.choice(X_train.shape[0])
        self.centroids = [X_train[index]]
        for _ in range(self.k-1):
            dists = np.sum([self.euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            dists /= np.sum(dists)
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]

        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [np.random.uniform(min_, max_) for _ in range(self.k)]

        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any():
            sorted_points = [[] for _ in range(self.k)]
            for x in X_train:
                dists = self.euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
        self.clusters = sorted_points
        return self.centroids, self.clusters
    
    def objective(self):
        clustering_objective = 0 
        for idx in range(self.k):
            centroid = self.centroids[idx]
            cluster = self.clusters[idx]
            for X in cluster:
                clustering_objective += np.sum((X - centroid)**2)

        return clustering_objective
    
    def accuracy(self):
        distribution_means = [[-1, -1], [1, -1], [0, 1]]
        labels = ['a', 'b', 'c']
        label_dict = {}

        for i in range(len(self.centroids)):
            centroid = self.centroids[i]
            dist = self.euclidean(centroid, distribution_means)
            label_idx = np.argmin(dist)
            label_dict[i] = labels[label_idx]

        correct = 0
        
        for i in range(len(self.X)):
            x = self.X[i]
            label = 'a'
            if i >= 100: label = 'b'
            if i >= 200: label = 'c'

            dist = self.euclidean(x, self.centroids)
            label_idx = np.argmin(dist)
            prediction = label_dict[label_idx]

            if label == prediction : correct += 1

        return correct/len(self.X)
