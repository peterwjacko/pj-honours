#%%
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# %%
# import data
objectFeatures = pd.read_csv('D:\Dropbox\Honours\Peter_Woodfordia\Data\ALLROI_FS_Merged.csv')
# %%
# rename columns
'''objectFeatures.columns = ['ROI',
                        'Set',
                        'class',
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
'''
objectFeatures.head()
# %%
# create subset of labelled objects
objectsLabelled = objectFeatures[objectFeatures["Class"].notna()] 

# create subset of unlabelled objects
objectsUnlabelled = objectFeatures[objectFeatures["Class"].isna()] 
# %%
#objectsUnlabelled.head()
# %%
# split data into train and test sets
####### drop/add object features here #######
X = objectsLabelled[objectsLabelled.columns[7:]] # x set are the features
y = objectsLabelled['Class'] # y set is the class labels (in this case species)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
# random_state argument can be set to make the split reproducable. Same value = same split. Remove random_state argument for random each time.
# stratify=y will split each class randomly with desired proportion stated in the test_size argument
# %%
#y_test.head() # view head of dataset to see if it worked
#y_train.head()
#X_test.head()
#X_train.head()

#y_test.shape # view shape of data sets
#y_train.shape
#X_test.shape
#X_train.shape
# %%
# scale/transform data here. https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = Normalizer()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%
# train and fit model
clf_SVC = svm.SVC(random_state=0).fit(X_train_scaled,y_train)

y_pred = clf_SVC.predict(X_test)

print("Train score: " + str(clf_SVC.score(X_train_scaled, y_train)))
print("Test score: " + str(clf_SVC.score(X_test_scaled, y_test)))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# %%
# find the best parameters for this model.

# Define a list of parameters for the models
params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    #,'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    #} 
## can add kernal to the testing parameters but can't plot in heat map below.

# We can build Grid Search model using the above parameters. 
# cv=5 means cross validation with 5 folds
grid_search = GridSearchCV(svm.SVC(random_state=0), params, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Train score:" + str(grid_search.score(X_train_scaled, y_train)))
print("Test score:" + str(grid_search.score(X_test_scaled, y_test)))
# %%
# print best parameters
print(grid_search.best_params_)

# %%
# plot best parameters on a heat map
results_df = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results_df.mean_test_score).reshape(6, 6)
# %%
sns.heatmap(scores, 
            annot=True, 
            xticklabels=params['gamma'], 
            yticklabels=params['C'])
# %%
# fit model again with optimal parameters and check accuracy
clf_SVC = svm.SVC(C=100, gamma=0.1, kernel='rbf', random_state=0).fit(X_train_scaled,y_train)

y_pred = clf_SVC.predict(X_test)

print('Kernel: RBF')
print("Train score: " + str(clf_SVC.score(X_train_scaled, y_train)))
print("Test score: " + str(clf_SVC.score(X_test_scaled, y_test)))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# %%
# TODO: finsih up the feature selection process for SVM

# Create a feature-selection transform, a scaler and an instance of SVM that we
# combine together to have an full-blown estimator
clf = Pipeline([('anova', SelectPercentile(chi2)),
                ('scaler', MinMaxScaler()),
                ('svc', svm.SVC(C=100, gamma=0.1, kernel='rbf', random_state=0))])

# %%
# Plot the cross-validation score as a function of percentile of features
score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    this_scores = cross_val_score(clf, X, y)
    score_means.append(this_scores.mean()) # TODO: getting NaN outputs??
    score_stds.append(this_scores.std())

plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title(
    'Performance of the SVM-Anova varying the percentile of features selected')
#plt.xticks(np.linspace(0, 100, 11, endpoint=True))
plt.xlabel('Percentile')
plt.ylabel('Accuracy Score')
plt.axis('tight')
plt.show()

# %%
# Generate confusion matrix
matrix = metrics.plot_confusion_matrix(clf_SVC, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show(matrix)
plt.show()

# %%
