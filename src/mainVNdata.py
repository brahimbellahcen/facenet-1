
from os.path import isdir
from os import listdir
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

import CONSTANTS
from extract_features import get_embedding
from face_classification import train_SVM


# load faces from dataset
# save to save_dir as npz format
# X, y : X[i] list of np array (faces) of person y[i]
def load_faces_from_dataset(data_dir, save_dir):
    print('load faces from dataset')
    X, y = list(), list()
    # tmp = 0
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

        # tmp += 1
        # if tmp > 4:
        #     break

    X = np.asarray(X)
    y = np.asarray(y)
    print(X.shape, y.shape)
    np.savez_compressed(save_dir, X, y)
    print('saved faces to ', save_dir)
    
    return X, y


#  extract feature
def convert_emb(npz_file, save_dir, model):
    data = np.load(npz_file, allow_pickle=True)
    X, y = data['arr_0'], data['arr_1']
    print('loaded data: ', X.shape, y.shape)

    print ('extracting features: ........................')
    X_emb = list()
    # per is faces list of one person)
    for per in X:
        faces = list()
        for face in per:
            emb = get_embedding(model, face)
            faces.append(emb)
        print('%d faces done!' %(len(faces)))

        X_emb.append(faces)
    X_emb = np.asarray(X_emb)
    print(X_emb.shape, y.shape)
    np.savez_compressed(save_dir, X_emb, y)
    print('save embeddings to ', save_dir)
    return X_emb, y



def split_data(npz_file, train_size, save_dir):
    print('split train test')
    data = np.load(npz_file, allow_pickle=True)
    X, y = data['arr_0'], data['arr_1']
    X_train, y_train, X_test, y_test = list(),list(), list(), list()

    for i in range(X.shape[0]):
        length = len(X[i])
        if length <= train_size:
            X_train.extend(X[i])
            y_train.extend(np.full((length,), y[i]))
        else:
            X_train.extend(X[i][:train_size])
            X_test.extend(X[i][train_size:])
            y_train.extend(np.full((train_size, ), y[i]))
            y_test.extend(np.full((length - train_size,), y[i]))

    X_train, y_train, X_test, y_test = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    np.savez_compressed(save_dir, X_train, y_train, X_test, y_test)
    print('saved train, test embeddings to ', save_dir)
    return X_train, y_train, X_test, y_test
        
        


def main():
    data_dir = CONSTANTS.root + '/data/VNCeleb-dataset/'

    load_faces_from_dataset(
        CONSTANTS.VNcelebdataset + '/',
        data_dir + 'detected_X_y.npz')
    
    facenet = load_model(CONSTANTS.root + '/model/facenet_keras.h5')

    convert_emb(data_dir + 'detected_X_y.npz', data_dir +'embeddings_X_y.npz', facenet)

    split_data(data_dir +'embeddings_X_y.npz', 10, data_dir + 'embeddings_Xtr_ytr_Xt_yt.npz')
    

    train_SVM(
        data_dir + 'embeddings_Xtr_ytr_Xt_yt.npz',
        CONSTANTS.root + '/model/SVM_FR_VNCeleb.joblib')
        

main()