import numpy as np

class PCA:
    def __init__(self, X):
        self.X = X.T
        self.U = None
        self.V = None
        self.S = None
        self.make_pca_space()

    def make_pca_space(self):
        '''
            make pca space with SVD.
        '''
        X = self.X - np.mean(self.X, axis=0)
        U, S, V = np.linalg.svd(X, full_matrices=False)
        U, V = svd_flip(U, V)
        self.U, self.S, self.V = U, S, V
        explained_variance_ = (S ** 2) / (self.X.shape[1] - 1)
        total_var = explained_variance_.sum()
        self.explained_variance_ratio = explained_variance_ / total_var

    def transform(self, X,  k):
        '''
            return image of X on first k
            principle components.
        '''
        if self.U is None:
            self.make_pca_space()
        return np.matmul(X, self.U[:, :k])

    def pov(self, k):
        '''
            return Poportion Of Variance for
            first k principle components.
        '''
        return self.explained_variance_ratio[k]

def svd_flip(u, v):
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v
