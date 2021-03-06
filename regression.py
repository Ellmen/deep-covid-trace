import numpy as np
from scipy.io import loadmat, savemat
from random import Random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

   
data = loadmat('./data/balanced_variable_seqs.mat')
X = data['X'] 
Y = data['Y'] 
X=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
Y= np.argmax(Y, axis=1)
seed = 1
# Random(seed).shuffle(X)
# Random(seed).shuffle(Y)
np.random.seed(seed)
np.random.shuffle(X)
np.random.seed(seed)
np.random.shuffle(Y)
splt = int(len(Y) * 0.90)
x_train = X[:splt]
y_train = Y[:splt]
x_test = X[splt:]
y_test = Y[splt:]



# # Define a pipeline to search for the best combination of PCA truncation
# # and classifier regularization.
# pca = PCA()
# # set the tolerance to a large value to make the example faster
# logistic = LogisticRegression(max_iter=10000)
# pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# X_digits=x_train
# y_digits =y_train

# # Parameters of pipelines can be set using ‘__’ separated parameter names:
# param_grid = {
#    # 'pca__n_components': [3, 15, 30, 70, 100, 120],
#    'pca__n_components': [120],
#    'logistic__C': np.logspace(-4, 4,4),
# }

# search = GridSearchCV(pipe, param_grid, n_jobs=-1,scoring='accuracy')
# search.fit(X_digits, y_digits)
# print("Best parameter (score=%0.3f):" % search.best_score_)
# print(search.best_params_)

# # Plot the PCA spectrum
# pca.fit(X_digits)

# fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
# ax0.plot(np.arange(1, pca.n_components_ + 1),
#           np.cumsum(pca.explained_variance_ratio_), 'x', linewidth=2)
# ax0.set_ylabel('Cumulative variance (%)')

# ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
#             linestyle=':', label='selected n_components')
# ax0.legend(prop=dict(size=12))

# # For each number of components, find the best classifier results
# results = pd.DataFrame(search.cv_results_)
# components_col = 'param_pca__n_components'
# best_clfs = results.groupby(components_col).apply(
#     lambda g: g.nlargest(1, 'mean_test_score'))

# best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
#                legend=False, ax=ax1)
# ax1.set_ylabel('accuracy')
# ax1.set_xlabel('n_components')

# plt.xlim(-1, 130)

# plt.tight_layout()
# plt.show()


#implementing the best model on test set using the best result
pca = PCA(n_components=120,  random_state=0)
pca.fit(x_train)
X_t_train = pca.transform(x_train)
X_t_test = pca.transform(x_test)

# Logistic regression
clf = LogisticRegression(max_iter=10000, C=21.5)
# clf.fit(X_t_train, y_train)
clf.fit(x_train, y_train)
# print ('score', clf.score(X_t_test, y_test))
print ('score', clf.score(x_test, y_test))
# print ('pred label', clf.predict(X_t_test))
