import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump
from time import time


def train_SVM(path_data, savedir):
    # path_data : path to file embbedings (Xtrain, ytrain, xtest, ytest.npz)


    print("training SVM ...................")
    #load face embeddings
    data = np.load(path_data)
    X_train, y_train, X_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    # nomalize input vectors
    in_encoder = Normalizer(norm='l2')
    X_train = in_encoder.transform(X_train)
    X_test = in_encoder.transform(X_test)

    # label encode
    out_encoder = LabelEncoder()
    out_encoder.fit(y_train)
    y_train = out_encoder.transform(y_train)
    y_test = out_encoder.transform(y_test)

    #GridSearchCV
    # print("Fitting the classifier to the training set")
    # t0 = time()
    # param_grid = {'C': [10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
    #             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # model = GridSearchCV(
    #     SVC(kernel='rbf', class_weight='balanced'), param_grid, n_jobs=-1
    # )
    # model = model.fit(X_train, y_train)
    # print("done in %0.3f minutes" % ((time() - t0)/60))
    # print("Best estimator found by grid search:")
    # print(model.best_estimator_)


    # # fit model
    print('fitting model')
    t0 = time()
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    print("done in %0.3f minutes" % ((time() - t0)/60))
    dump(model, savedir)

    # evaluate
    print('evaluating......')
    t0 = time()
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    score_train = accuracy_score(y_train, y_train_pred)
    socre_test = accuracy_score(y_test, y_test_pred)
    print("done in %0.3f minutes" % ((time() - t0)/60))
    print('Acccuracy: train = %.3f, test = %.3f' % (score_train*100, socre_test*100))


