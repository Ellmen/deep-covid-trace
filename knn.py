import numpy as np
from scipy.io import loadmat, savemat
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = loadmat('./data/balanced_variable_seqs.mat')
X = data['X'] 
Y = data['Y'] 
X=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
seed = 1
np.random.seed(seed)
np.random.shuffle(X)
np.random.seed(seed)
np.random.shuffle(Y)
splt = int(len(Y) * 0.9)
x_train = X[:splt]
y_train = Y[:splt]
x_test = X[splt:]
y_test = Y[splt:]


clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
print ('score', clf.score(x_test, y_test))
