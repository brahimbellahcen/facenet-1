import numpy as np
from tensorflow.keras.models import load_model
from os import listdir

import _constants
from detect import extract_faces_from_dataset
from extract_features import get_embedding
from face_classification import train_SVM

#extract all face, label from dataset and save data as npz
def extract_faces(train_dir, test_dir, save_dir):
    X_train, y_train = extract_faces_from_dataset(train_dir)
    print('X_train: %s, y_train: %s' % (str(X_train.shape), str(y_train.shape)))

    X_test, y_test = extract_faces_from_dataset(test_dir)
    print('X_test: %s, y_test: %s' % (str(X_test.shape), str(y_test.shape)))

    np.savez_compressed(save_dir, X_train, y_train, X_test, y_test)
    print('saved extracted faces to ', save_dir)


# cover faces to embebings and save as npz
# file: detected......npz file
def cover_to_embeddings(file, model, save_dir):
    #load dataset
    data = np.load(file)
    X_train, y_train, X_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('loaded data: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    
    #covert each face in the train set to an embedding
    X_train_emb = list()
    print('extracting training set .....................')
    for face_pixels in X_train:
        embedding = get_embedding(model, face_pixels)
        X_train_emb.append(embedding)
    X_train_emb = np.asarray(X_train_emb)
    print('X_train_emb shape: ', X_train_emb.shape)

    #covert (extract)  test
    print('extracting test set......................')
    X_test_emb = list()
    for face_pixels in X_test:
        embedding = get_embedding(model, face_pixels)
        X_test_emb.append(embedding)
    X_test_emb = np.asarray(X_test_emb)
    print('X_test_emb shape: ', X_test_emb.shape)

    #save
    np.savez_compressed(save_dir, X_train_emb, y_train, X_test_emb, y_test)
    print('saved embeddings to ', save_dir)


def mainx():
    root = _constants.root
    dataset = root + '/data/5-celebrity-faces-dataset/'
    extract_faces(dataset + 'train/', dataset + 'val/', dataset + 'detected_Xtr_ytr_Xt_yt.npz')

    facenet = load_model(root + '/model/facenet_keras.h5')
    cover_to_embeddings(dataset + 'detected_Xtr_ytr_Xt_yt.npz', model=facenet,\
        save_dir=dataset + 'embeddings_Xtr_ytr_Xt_yt.npz')
    

    train_SVM(dataset + 'embeddings_Xtr_ytr_Xt_yt.npz', root + '/model/SVM_FR_5celeb.joblib')
    print('Done!')

#run
mainx()
