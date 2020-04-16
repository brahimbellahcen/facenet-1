from os import listdir
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN


#extract a single face
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



