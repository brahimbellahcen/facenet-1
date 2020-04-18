import numpy as np
from tensorflow.keras.models import load_model

# extract the face embbedding for one face
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')

    #standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    #transform face into one sample (160, 160, 3) -> (1, 160, 160, 3)
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)

    return yhat[0]
    

# #load dataset
# print('loading data.......')
# data = np.load('data/processed_data_Xtrain_ytrain_Xtest_ytest.npz')
# X_train, y_train, X_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# print('loaded data: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# #load the facenet model
# print('loading the facenet model........')
# model = load_model('model/facenet_keras.h5')
# print('loaded model!')

# #covert each face in the train set to an embedding
# X_train_emb = list()
# print('extracting training set .....................')
# for face_pixels in X_train:
#     embedding = get_embedding(model, face_pixels)
#     X_train_emb.append(embedding)
# print('extracted training set!')
# X_train_emb = np.asarray(X_train_emb)
# print('X_train_emb shape: ', X_train_emb.shape)

# #covert (extract)  test
# print('extracting test set......................')
# X_test_emb = list()
# for face_pixels in X_test:
#     embedding = get_embedding(model, face_pixels)
#     X_test_emb.append(embedding)
# print('extracted_testset!')
# X_test_emb = np.asarray(X_test_emb)
# print('X_test_emb shape: ', X_test_emb.shape)

# #save
# np.savez_compressed('data/embeddings_Xtr_ytr_Xt_yt.npz', X_train_emb, y_train, X_test_emb, y_test)

