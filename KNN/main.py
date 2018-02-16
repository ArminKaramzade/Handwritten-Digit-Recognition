import utils, knn, pca
import numpy as np
import timeit
start_time = timeit.default_timer()
n_component = 54
k = 3

digits_train, y_train = utils.read('train')
digits_test, y_test = utils.read('test')

print('Extracting features ...')
digits_train, digits_test = utils.resize(digits_train, 16), utils.resize(digits_test, 16)
digits_train, digits_test = utils.get_deskew_imgs(digits_train), utils.get_deskew_imgs(digits_test)
holes_train, holes_test = utils.get_hole_features(digits_train), utils.get_hole_features(digits_test)
pix_train, pix_test = utils.get_pix_features(digits_train), utils.get_pix_features(digits_test)
X_train, X_test = np.hstack([pix_train, holes_train]), np.hstack([pix_test, holes_test])

mean_normalizer = utils.normalization(X_train)
X_train = mean_normalizer.transform(X_train)
X_test = mean_normalizer.transform(X_test)

pca = pca.PCA(X_train)
X_train_reduced = pca.transform(X_train, n_component)
X_test_reduced = pca.transform(X_test, n_component)

clf = knn.KNN(mode='weighted')
print('Fitting ...')
clf.fit(X_train_reduced, y_train)
print('Evaluating ...')
print("Accuracy on test data:", clf.score(X_test_reduced, y_test, k))
elapsed = timeit.default_timer() - start_time
print("Execution time: " + str(int(elapsed/60)) + ":" + str(int(elapsed % 60)))
