import numpy as np
import os, struct

def read(dataset, path = "../data/"):
    '''
        read MNIST data.
        dataset -> "train" or "test"
        path -> path to MNIST dataset directory
    '''
    name = ""
    if dataset == "test":
        name = "t10k-"
    elif dataset == "train":
        name = "train-"
    else:
        raise (ValueError, "dataset in read() must be 'train' or 'test'")
    img_name = os.path.join(path, name + 'images-idx3-ubyte')
    lbl_name = os.path.join(path, name + 'labels-idx1-ubyte')
    with open(lbl_name, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    with open(img_name, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    X = []
    y = []
    for i in range(len(lbl)):
        X.append(img[i])
        y.append(lbl[i])
    return (np.array(X), np.array(y))

def get_deskew_imgs(digits):
    import cv2
    def deskew(img):
        SZ = img.shape[0]
        affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
        return img
    imgs = []
    for digit in digits:
        imgs.append(deskew(digit))
    return np.array(imgs)

def get_pix_features(digits):
    n, w, h = digits.shape
    return digits.reshape((n, w*h)).astype('float64')

def get_haar_features(digits, type='type-2-x'):
    from skimage.feature import haar_like_feature
    X = []
    _, w, h = digits.shape
    for digit in digits:
        X.append(haar_like_feature(digit, 0, 0, w, h, feature_type=type))
    return np.array(X)

def get_hole_features(digits):
    def dfs(g):
        dr = [1, 0, 0, -1, 1, 1, -1, -1]
        dc = [0, -1, 1, 0, 1, -1, 1, -1]
        def _in(x, y):
            return 0 <= x and x < g.shape[0] and 0 <= y and y < g.shape[1]
        def _dfs_util(i, j, mark):
            if mark[i][j]:
                return
            mark[i][j] = 1
            for k in range(8):
                if _in(i + dr[k], j + dc[k]) and g[i+dr[k], j+dc[k]] == 0:
                    _dfs_util(i+dr[k], j+dc[k], mark)
        ret = 0
        mark = np.zeros((g.shape[0], g.shape[1]))
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if g[i][j] == 0 and mark[i][j] == 0:
                    _dfs_util(i, j, mark)
                    ret += 1
        return ret
    X = []
    for digit in digits:
        X.append(dfs(digit))
    return np.array(X).reshape((len(X), 1))

def resize(digits, size=16):
    import cv2
    resized = []
    for digit in digits:
        resized.append(cv2.resize(digit, (size, size), interpolation=cv2.INTER_AREA))
    return np.array(resized)

class normalization:
    def __init__(self, X_train):
        from sklearn import preprocessing
        self.normalizer = preprocessing.MinMaxScaler()
        self.normalizer.fit(X_train)
    def transform(self, X):
        return self.normalizer.transform(X)

def get_hog_feature(digits):
    import cv2
    def hog_feature(img):
        winSize = (16,16)
        blockSize = (8,8)
        blockStride = (4,4)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        discriptor = hog.compute(img)
        return discriptor.reshape((discriptor.shape[0]))
    X = []
    for digit in digits:
        X.append(hog_feature(digit))
    return np.array(X)

def get_circle_features(digits):
    from math import sqrt
    def circle_filter(img):
        n = 8
        m = 4
        filtered = [[0. for _ in range(m)] for _ in range(n)]
        for j in range(m):
            R = float(len(img) / (2 * m))*(j+1)
            for x in range(len(img)):
                for y in range(len(img[0])):
                    X = abs(7.5 - x) + 0.5
                    Y = abs(7.5 - y) + 0.5
                    mid = len(img)/2
                    w = 0
                    if X * X + Y * Y <= R * R: w = 1
                    else:
                        if (X - 1) * (X - 1) + (Y - 1) * (Y - 1) <= R * R: w = (sqrt(X*X + Y*Y) - R)/1.41
                        elif X == 0 and Y - 1 <= R: w = Y-R
                        elif Y == 0 and X - 1 <= R: w = X-R
                    if x < mid and y < mid:  # T[0], T[7]
                        if x < y:  # T[0]
                            filtered[0][j] += img[x][y] * w
                        elif x == y:  # T[0], T[7]
                            filtered[0][j] += img[x][y] * w
                            filtered[7][j] += img[x][y] * w
                        elif x > y:  # T[7]
                            filtered[7][j] += img[x][y] * w
                    elif x >= mid and y < mid:  # T[5], T[6]
                        if (15 - x) < y:  # T[5]
                            filtered[5][j] += img[x][y] * w
                        elif (15 - x) == y:  # T[5], T[6]
                            filtered[5][j] += img[x][y] * w
                            filtered[6][j] += img[x][y] * w
                        elif (15 - x) > y:  # T[6]
                            filtered[6][j] += img[x][y] * w
                    elif x < mid and y >= mid:  # T[1], T[2]
                        if x < (15 - y):  # T[1]
                            filtered[1][j] += img[x][y] * w
                        elif x == (15 - y):  # T[1], T[2]
                            filtered[1][j] += img[x][y] * w
                            filtered[2][j] += img[x][y] * w
                        elif x > (15 - y):  # T[2]
                            filtered[2][j] += img[x][y] * w
                    if x >= mid and y >= mid:  # T[3], T[4]
                        if (15 - x) < (15 - y):  # T[4]
                            filtered[4][j] += img[x][y] * w
                        elif (15 - x) == (15 - y):  # T[4], T[3]
                            filtered[3][j] += img[x][y] * w
                            filtered[4][j] += img[x][y] * w
                        elif (15 - x) > (15 - y):  # T[3]
                            filtered[3][j] += img[x][y] * w
        for i in range(n):
            for j in range(m-1, 0, -1):
                filtered[i][j] -= filtered[i][j-1]
        return np.array(filtered).reshape((32))
    X = []
    for digit in digits:
        X.append(circle_filter(digit))
    return np.array(X)
