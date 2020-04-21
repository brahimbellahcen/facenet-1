
from os.path import isdir
from os import listdir
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

import CONSTANTS
from extract_features import get_embedding
from main5Celeb import convert_to_embeddings
from face_classification import train_SVM


def load_face(data_dir, save_dir, train_size):
    X_train, y_train, X_test, y_test = list(),list(), list(), list()
    X, y = list(), list()
    tmp = 0
    for subdir in listdir(data_dir):
        faces = list() #save faces from subdir
        path = data_dir + subdir + '/'
        #skip file is not dir
        if not isdir(path):
            continue

        for item in listdir(path):
            file = path + item
            img = Image.open(file)
            img = img.convert('RGB')
            img = img.resize((160, 160))
            pixels = np.asarray(img)
            faces.append(pixels)
            # list_faces_of_person (np.array_face)
        
        X.append(faces)  #list_of_persons( list (nparray))
        y.append(subdir)

        tmp += 1
        if tmp > 50:
            break
        
    #split traint, test
    for i in range(len(X)):
        person = X[i]
        number_of_faces = len(person)
        person = np.asarray(person)
        if number_of_faces <= train_size:
            continue
        person = np.asarray(person)
        X_train.extend(person[:train_size])
        X_test.extend(person[train_size:])
        y_train.extend(np.full( (train_size,), y[i] ))
        y_test.extend(np.full( (number_of_faces-train_size,), y[i] ))
            
    X_train, y_train, X_test, y_test \
    = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    np.savez_compressed(save_dir, X_train, y_train, X_test, y_test)
    print('saved extracted faces to ', save_dir)
    return X_train, y_train, X_test, y_test

def mainx():
    data_dir = CONSTANTS.root + '/data/VNCeleb-dataset/'
    load_face(
        CONSTANTS.VNcelebdataset + '/',
        data_dir + 'detected_Xtr_ytr_Xt_yt.npz',
        train_size=10)
    
    facenet = load_model(CONSTANTS.root + '/model/facenet_keras.h5')
    
    convert_to_embeddings(
        data_dir + 'detected_Xtr_ytr_Xt_yt.npz',
        facenet,
        data_dir + 'embeddings_Xtr_ytr_Xt_yt.npz'
    )
    
    train_SVM(
        data_dir + 'embeddings_Xtr_ytr_Xt_yt.npz',
        CONSTANTS.root + '/model/SVM_FR_VNCeleb.joblib')

# mainx()