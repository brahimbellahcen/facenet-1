import CONSTANTS
import numpy as np


def split_data(npz_file, train_size, save_dir):
    print('split train test')
    data = np.load(npz_file, allow_pickle=True)
    X, y = data['arr_0'], data['arr_1']
    X_train, y_train, X_test, y_test = list(),list(), list(), list()

    for i in range(100):
    # for i in range(X.shape[0]):
        length = len(X[i])
        if length <= 5:
            # X_train.extend(X[i])
            # y_train.extend(np.full((length,), y[i]))
            pass
        else:
            ltrain = int(length * train_size)
            X_train.extend(X[i][:ltrain])
            X_test.extend(X[i][ltrain:])
            y_train.extend(np.full((ltrain, ), y[i]))
            y_test.extend(np.full((length - ltrain,), y[i]))

    X_train, y_train, X_test, y_test = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    np.savez_compressed(save_dir, X_train, y_train, X_test, y_test)
    print('saved train, test embeddings to ', save_dir)
    return X_train, y_train, X_test, y_test

data_dir = CONSTANTS.root + '/data/VNCeleb-dataset/'
split_data(data_dir +'embeddings_X_y.npz', 0.8, data_dir + 'embeddings_Xtr_ytr_Xt_yt_6_100.npz')
