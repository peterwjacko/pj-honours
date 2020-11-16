#%%
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
# import data
objectFeatures = pd.read_csv('D:\ProgrammingProjects\honours\ROI_C_Features.csv', delimiter=';')
# %%
objectFeatures.columns = ['class',
                'meanG', 
                'stdG', 
                'meanR', 
                'stdR', 
                'meanCHM',
                'stdCHM',
                'meanVARI',
                'compactness',
                'asymmetry',
                'roundness',
                'meanGLCM',
                'stdGLCM']
# %%
#train, test = train_test_split(objectFeatures, test_size=.25, train_size=.75)
X = objectFeatures[objectFeatures.columns[1:]]
y = objectFeatures['class'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
# %%
y_test
# %%
'''X_train = train[train.columns[1:]]
y_train = train['class']
X_test = test[test.columns[1:]] 
y_test = test['class']'''
# %%
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 # %%
svc_model = svm.SVC(random_state=0).fit(X_train_scaled,y_train)
print("train score - " + str(svc_model.score(X_train_scaled, y_train)))
print("test score - " + str(svc_model.score(X_test_scaled, y_test)))
# %%
svc_model

# %%
#We can use a grid search to find the best parameters for this model. Lets try

#Define a list of parameters for the models
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

#We can build Grid Search model using the above parameters. 
#cv=5 means cross validation with 5 folds
grid_search = GridSearchCV(svm.SVC(random_state=0), params, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("train score - " + str(grid_search.score(X_train_scaled, y_train)))
print("test score - " + str(grid_search.score(X_test_scaled, y_test)))

# %%
print(grid_search.best_params_)

# %%
results_df = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results_df.mean_test_score).reshape(6, 6)
# %%
sns.heatmap(scores, annot=True, 
            xticklabels=params['gamma'], yticklabels=params['C'])
# %%
pcaObjects = PCA(n_components=5)
pcObj = pcaObjects.fit_transform(objectFeatures[2:5])
# %%
