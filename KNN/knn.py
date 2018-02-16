import numpy as np
from sklearn.neighbors import KDTree


class KNN:
    def __init__(self, c=10, mode='weighted'):
        self.c = c
        self.kd_tree = None
        self.mode = mode

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.kd_tree = KDTree(X_train, leaf_size=10, metric='euclidean')

    def predict(self, X_test, k):
        y_pred = []
        for x in X_test:
            distances, indices = self.kd_tree.query(x.reshape(1, -1), k=k)
            distances = [1e-6 if d == 0 else d for d in distances[0]]
            count_list = [0 for _ in range(self.c)]
            for it, index in enumerate(indices[0]):
                count_list[self.y_train[index]] += 1 if self.mode == 'uniform' else (1. / distances[it])
            y_pred.append(np.argmax(count_list))
        return y_pred

    def score(self, X_test, y_test, k):
        y_pred = self.predict(X_test, k)
        true_num = len([1 for i in range(X_test.shape[0] ) if y_pred[i] == y_test[i]])
        accuracy = float(true_num) / X_test.shape[0] * 100
        return accuracy
