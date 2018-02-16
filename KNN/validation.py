import utils, knn, pca
from sklearn.model_selection import KFold
import timeit
import numpy as np


digits_train, y_train = utils.read('train')
digits_test, y_test = utils.read('test')

digits_train, digits_test = utils.resize(digits_train, 16), utils.resize(digits_test, 16)
digits_train, digits_test = utils.get_deskew_imgs(digits_train), utils.get_deskew_imgs(digits_test)
holes_train, holes_test = utils.get_hole_features(digits_train), utils.get_hole_features(digits_test)
pix_train, pix_test = utils.get_pix_features(digits_train), utils.get_pix_features(digits_test)
X_train, X_test = np.hstack([pix_train, holes_train]), np.hstack([pix_test, holes_test])

mean_normalizer = utils.normalization(X_train)
X_train = mean_normalizer.transform(X_train)
X_test = mean_normalizer.transform(X_test)

mx_score = 0
best = (-1, -1)
clf = knn.KNN(mode='weighted')
for n_component in range(3, 61, 3):
    for k in range(1, 11):
        _pca = pca.PCA(X_train)
        X_train_reduced = _pca.transform(X_train, n_component)
        X_test_reduced = _pca.transform(X_test, n_component)

        start_time = timeit.default_timer()
        validation_scores = []
        kf = KFold(n_splits=10)
        for t_idx, v_idx in kf.split(X_train_reduced):
            X_train_T, X_train_V = X_train_reduced[t_idx], X_train_reduced[v_idx]
            y_train_T, y_train_V = y_train[t_idx], y_train[v_idx]
            clf.fit(X_train_T, y_train_T)
            validation_score = clf.score(X_train_V, y_train_V, k)
            validation_scores.append(validation_score)
        avg_val_score = np.mean(validation_scores)
        if avg_val_score > mx_score:
            mx_score = avg_val_score
            best = (n_component, k)
        print("[n_components:", n_component, ",n_neighbours:", k, "] Average validation score: ", avg_val_score)
        print("Accuracy on test data:", clf.score(X_test_reduced, y_test, k))
        elapsed = timeit.default_timer() - start_time
        print("Execution time: " + str(int(elapsed/60)) + ":" + str(int(elapsed % 60)))
        print('------------------------')

print("=========================================")
print("best n_component:", best[0], "best n_neighbours:", best[1])
print("=========================================")
