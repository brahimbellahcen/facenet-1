from face_classification import train_SVM
import CONSTANTS


train_SVM (CONSTANTS.root +'/data/VNCeleb-dataset/embeddings_Xtr_ytr_Xt_yt.npz',
             CONSTANTS.root + '/model/SVM_FR_VNCeleb_train2.joblib')