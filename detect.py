from os import listdir
from os.path import isdir
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN


#extract a single face
#return a numpy array 2X2, pixels of face
def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)

    detector = MTCNN()
    results = detector.detect_faces(pixels)
    
    #first face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    #extract face
    pixels_face = pixels[y1:y2, x1:x2]
    image_face = Image.fromarray(pixels_face)
    image_face = image_face.resize(required_size)
    pixels_face_result = np.asarray(image_face)

    return pixels_face_result




# load and extract faces for all images in a directory
def load_faces(dir):
    faces = list()
    for filename in listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)
    
    return faces


# load a dataset that contains one subdir for each class
# each subdir contains images of one person
#
def load_dataset(dir):
    X, y = list(), list()
    for subdir in listdir(dir):
        path = dir + subdir + '/'
        #skip file is not dir
        if not isdir(path):
            continue
        
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        #summarize
        print('==> loaded %d examples for class: %s' % (len(faces), subdir))
        #store
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


# load train dataset
X_train, y_train = load_dataset('data/train/')
print(X_train.shape, y_train.shape)

#load test dataset
X_test, y_test = load_dataset('data/val/')
print(X_test.shape, y_test.shape)

#save to compressed file
np.savez_compressed('data/processed_data_Xtrain_ytrain_Xtest_ytest.npz', \
    X_train, y_train, X_test, y_test)