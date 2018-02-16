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

def purity(clusters, y, k):
    n = y.shape[0]
    classes = np.unique(y).shape[0]
    p = 0.0
    for i in range(k):
        idx = np.where(clusters==i)[0]
        tmp = np.zeros(classes)
        for j in range(idx.shape[0]):
            tmp[y[idx[j]]] += 1
        mx = 0
        for j in range(classes):
            mx = max(mx, tmp[j])
        p += mx
    return p / n

def rand_index(clusters, y, k):
    n = y.shape[0]
    classes = np.unique(y).shape[0]
    tp, tn, fp, fn = 0., 0., 0., 0.
    positives, negatives = 0, 0

    c_list, idx_list, tmp_list = [], [], []
    for i in range(k):
        idx = np.where(clusters==i)[0]
        ci = clusters[idx]
        c_list.append(ci)
        idx_list.append(idx)
        tmp = np.zeros(classes)
        for j in range(idx.shape[0]):
            tmp[y[idx[j]]] += 1
        tmp_list.append(tmp)

    for i in range(k):
        for j in range(i+1, k):
            negatives += c_list[i].shape[0] * c_list[j].shape[0]
            for l in range(classes):
                fn += tmp_list[i][l] * tmp_list[j][l]
        positives += c_list[i].shape[0] * (c_list[i].shape[0]-1) / 2
        for j in range(classes):
            tp += tmp_list[i][j] * (tmp_list[i][j]-1) / 2
    tn = negatives - fn
    fp = positives - tp
    return (tp + tn) / (tp + tn + fp + fn)

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

def normalize(X):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(X)
