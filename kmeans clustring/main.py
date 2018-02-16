import kmeans, utils, pca
import numpy as np
import timeit

pov = .98
classes = 10
iter = 15
#-------------
# reading
digits, y = utils.read("train")
idx = np.random.permutation(y.shape[0])
digits = digits[idx, :]
y = y[idx]

#-------------
# feature extraction
digits = utils.resize(digits, 16)
digits = utils.get_deskew_imgs(digits)
# pix = utils.get_pix_features(digits)
hog = utils.get_hog_feature(digits)
# X = np.hstack([pix])
X = np.hstack([hog])
X = utils.normalize(X)

#-------------
# data reduction
pca = pca.PCA(X)
n_component = 0
gained = 0
while(gained < pov):
    gained += pca.pov(n_component)
    n_component += 1
X_reduced = pca.transform(X, n_component)

#-------------
# clustring
start_time = timeit.default_timer()
print("clustring with k-means with k =", classes, "...")
k_means = kmeans.kmeans(X_reduced, y)
clusters, centers = k_means.train(classes, iter)
print("K-means purity:", utils.purity(clusters, y, classes))
print("K-means rand-index:", utils.rand_index(clusters, y, classes))
elapsed = timeit.default_timer() - start_time
print("execution time: ", str(int((elapsed/60)))+':'+str(int(elapsed%60)))
