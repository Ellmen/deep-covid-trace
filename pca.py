import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from random import Random
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA


#convert_data()
data = loadmat('balanced_variable_seqs.mat')
X = data['X'] 
Y = data['Y'] 
X=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
Y= np.argmax(Y, axis=1)
seed = 1
Random(seed).shuffle(X)
Random(seed).shuffle(Y)
# splt = int(len(Y) * 0.9)
# x_train = X[:splt]
# y_train = Y[:splt]
# x_test = X[splt:]
# y_test = Y[splt:]

# #visualizing in 2D
# pca = PCA(n_components=2,  random_state=22) #2-dimensional PCA
# transformed = pd.DataFrame(pca.fit_transform(X))

# plt.scatter(transformed[Y==0][0], transformed[Y==0][1], label='Class 0', c='purple')
# plt.scatter(transformed[Y==1][0], transformed[Y==1][1], label='Class 1', c='orange')
# plt.scatter(transformed[Y==2][0], transformed[Y==2][1], label='Class 2', c='red')
# plt.scatter(transformed[Y==3][0], transformed[Y==3][1], label='Class 3', c='lightgreen')
# plt.scatter(transformed[Y==4][0], transformed[Y==4][1], label='Class 4', c='darkblue')
# plt.scatter(transformed[Y==5][0], transformed[Y==5][1], label='Class 5', c='cyan')

# plt.xlabel('First principal component')
# plt.ylabel('Second Principal Component')
# plt.legend(loc='best')
# plt.show()

#visualizing in 3D
pca = PCA(n_components=3,  random_state=22) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(transformed[Y==0][0], transformed[Y==0][1],transformed[Y==0][2], label='Class 0', c='purple')
ax.scatter(transformed[Y==1][0], transformed[Y==1][1], transformed[Y==1][2], label='Class 1', c='orange')
ax.scatter(transformed[Y==2][0], transformed[Y==2][1], transformed[Y==2][2], label='Class 2', c='red')
ax.scatter(transformed[Y==3][0], transformed[Y==3][1], transformed[Y==3][2], label='Class 3', c='lightgreen')
ax.scatter(transformed[Y==4][0], transformed[Y==4][1], transformed[Y==4][2], label='Class 4', c='darkblue')
ax.scatter(transformed[Y==5][0], transformed[Y==5][1], transformed[Y==5][2], label='Class 5', c='cyan')

ax.set_xlabel('First principal component')
ax.set_ylabel('second Principal Component')
ax.set_zlabel('Third Principal Component')
plt.legend(loc='best')
plt.show()

