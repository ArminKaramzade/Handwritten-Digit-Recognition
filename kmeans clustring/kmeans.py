import numpy as np
import utils

class kmeans:
    def __init__(self, X, y):
        self.X = X
        self.tmp = y

    def train(self, k, m=10, ):
        best_y = None
        best_centers = None
        best_error = 1e10
        for i in range(m):
            y, centers = self.run(k)
            err = self.error(centers, y)
            if err < best_error:
                best_error = err
                best_y = y
                best_centers = centers
        return best_y, best_centers

    def run(self, k):
        centers = self.X[np.random.choice(self.X.shape[0], k, replace=False), :]
        finished = False
        y = None
        while not finished:
            y_new = self.assign(centers)
            if (y is not None) and (np.array_equal(y, y_new)):
                finished = True
            centers = self.get_centers(y_new, k)
            y = y_new
        return y, centers

    def assign(self, centers):
        y = np.zeros(self.X.shape[0], dtype=np.int32)
        for i in range(self.X.shape[0]):
            nearest = 1e6
            for j in range(centers.shape[0]):
                dist = np.linalg.norm(centers[j]-self.X[i, :])
                if dist < nearest:
                    nearest = dist
                    y[i] = j
        return y

    def get_centers(self, y, k):
        centers = np.zeros((k, self.X.shape[1]))
        num = np.zeros(k)
        for i in range(self.X.shape[0]):
            centers[y[i]] += self.X[i, :]
            num[y[i]] += 1
        for i in range(k):
            if num[i]:
                centers[i] = centers[i] / num[i]
        return centers

    def error(self, centers, y):
        err = 0
        for i in range(self.X.shape[0]):
            center = centers[y[i]]
            err += np.linalg.norm(center-self.X[i, :])
        return err
