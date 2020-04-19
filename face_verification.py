from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine
from detect import extract_face, load_faces, load_dataset
from extract_features import get_embedding
import matplotlib.pyplot as plt


# input is file:
def verification_file(file1, file2, model=None, plot=False, thresh=0.5):
    if model is None:
        model = load_model('model/facenet_keras.h5')

    #detect face
    face1 = extract_face(file1)
    face2 = extract_face(file2)

    return verification_facepixels(face1, face2, model=model, plot=plot, thresh=thresh)


# input is face_pixels
def verification_facepixels(face1, face2, model=None, plot=False, thresh=0.5):
    if model is None:
        model = load_model('model/facenet_keras.h5')


    #extract feature
    face1_emb = get_embedding(model, face1)
    face2_emb = get_embedding(model, face2)

    #matching
    score = cosine(face1_emb, face2_emb)
    title = None
    result = 0 #not match
    if score <= thresh:
        title = 'Match! Score = %.3f' % (score)
        result = 1
    else:
        title = 'NOT match! Score = %.3f' % (score)
        result = 0
    
    if plot:
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(face1)
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(face2)
        plt.suptitle(title)
        plt.show()
    else:
        print(title)

    return result

