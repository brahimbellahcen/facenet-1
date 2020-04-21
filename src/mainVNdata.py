
from os.path import isdir
from os import listdir
from PIL import Image
import numpy as np





def load_face(data_dir, save_dir, train_size):
    # X_train, y_train, X_test, y_test = list(),list(), list(), list()

    X, y = list(), list()
    for subdir in listdir(data_dir):
        faces = list() #save faces from subdir
        path = data_dir + subdir + '/'
        #skip file is not dir
        if not isdir(path):
            continue
        #skip person has  <= 10 images
        # if len(listdir(path)) <= train_size:
        #     continue

        for item in listdir(path):
            file = path + item
            img = Image.open(file)
            img = img.convert('RGB')
            # pixels = pixels.resize((160, 160))
            pixels = np.asarray(img)
            faces.append(pixels)
        
        faces = np.asarray(faces)
        X.append(faces)
        y.append(subdir)
