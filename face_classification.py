import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#load face embeddings
data = np.load('data/embeddings_Xtr_ytr_Xt_yt.npz')
X_train, y_train, X_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# nomalize input vectors
in_encoder = Normalizer(norm='l2')
X_train = in_encoder.transform(X_train)
X_test = in_encoder.transform(X_test)

# label encode
out_encoder = LabelEncoder()
out_encoder.fit(y_train)
y_train = out_encoder.transform(y_train)
y_test = out_encoder.transform(y_test)

# fit model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

score_train = accuracy_score(y_train, y_train_pred)
socre_test = accuracy_score(y_test, y_test_pred)

print('Acccuracy: train = %.3f, test = %.3f' % (score_train*100, socre_test*100))


