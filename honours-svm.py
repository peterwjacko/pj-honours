#%%
import random
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from datetime import datetime, date, time

# %%
# import data
objectFeatures = pd.read_csv('D:\Dropbox\Honours\Peter_Woodfordia\Data\ALLROI_FS_Merged.csv')

# %%
# define which column caintains the class labels

classLabel = "Species"

# %%
# create subset of labelled objects
objectsLabelled = objectFeatures[objectFeatures[classLabel].notna()] 

# create subset of unlabelled objects
objectsUnlabelled = objectFeatures[objectFeatures[classLabel].isna()] 

# %%
# drop features and classes

####### drop/add object features here #######
AllFeatures = [ 'Asymmetry', 'Border_index', 'Compactness', 'GLCM_Ang_2',
                'GLCM_Dissimilarity', 'GLCM_Entropy', 'GLCM_Homogeneity', 'GLCM_Mean',
                'GLCM_StdDev', 'Max_diff', 'Max_lidarCHM', 'Mean_blue', 'Mean_GLI',
                'Mean_green', 'Mean_GRVI', 'Mean_LidarCHM', 'Mean_red', 'Mean_VARI',
                'Mean_VVI', 'Roundness', 'Shape_index', 'Std_blue', 'Std_GLI',
                'Std_green', 'Std_GRVI', 'Std_lidarCHM', 'Std_red', 'Std_VARI',
                'Std_VVI']
                
'''featureSubset = ['Mean_LidarCHM',
                 'Max_lidarCHM',
                 'Mean_GLI',
                 'Max_diff',
                 'Std_lidarCHM',
                 'Std_VARI',
                 'Std_GRVI',
                 'Std_red',
                 'Std_green']'''
                 
#randomFeatures = random.sample(AllFeatures, 7)                 
                 
#featureSubset = randomFeatures
featureSubset = AllFeatures

# %%
####### drop classes here #######

# list classes to "drop" (relabel as non-target)
classSubset = ['Angophora subvelutina', 
               'Syncarpia verecunda', 
               'Eucalyptus grandis', 
               'Eucalyptus microcorys']

AllClasses = objectsLabelled[classLabel]
AllClasses = list(AllClasses.unique())
AllClasses.sort()

# this drops classes not in the list above by renaming them to "Non-target"
# this is because it is still useful that they are labelled to reduce comission error
objectsLabelled.loc[objectsLabelled[classLabel].isin(classSubset), ['Species', 'Genus', 'Family']] = "Non-target"
objectsLabelled = objectsLabelled.sort_values(by=[classLabel])
# %%
objectsLabelled
    
# %%
# split data into train and test sets

X = objectsLabelled[featureSubset] # x set are the features
y = objectsLabelled[classLabel] # y set is the class labels (in this case species)

uniqueLabel = objectsLabelled[classLabel]
uniqueLabel = list(uniqueLabel.unique()) # get unique labels/classes
uniqueLabel.sort() # sort in alphabetical

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
# random_state argument can be set to make the split reproducable. Same value = same split. Remove random_state argument for random each time.
# stratify=y will split each class randomly with desired proportion stated in the test_size argument
classFreqTrain = y_train.value_counts()
classFreqTest = y_test.value_counts()
featureList = list(X_train.columns)
featureList = pd.DataFrame(featureList)

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
# find the best parameters for this model.

# Define a list of parameters for the models
params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 150, 200, 300, 400, 600, 800, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    #'kernel': ['linear']
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    } 
## can add kernal to the testing parameters but can't plot in heat map below.

# We can build Grid Search model using the above parameters. 
# cv=5 means cross validation with 5 folds
grid_search = GridSearchCV(svm.SVC(random_state=0), params, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Train score:" + str(grid_search.score(X_train_scaled, y_train)))
print("Test score:" + str(grid_search.score(X_test_scaled, y_test)))
# %%
# print best parameters
grid_search.best_params_

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
# fit model with optimal parameters and test
hyperparamC = 200
hyperparamGamma = 0.1
hyperparamKernel = 'rbf'

clf_SVC = svm.SVC(C=hyperparamC, 
                  gamma=hyperparamGamma,
                  kernel=hyperparamKernel, 
                  random_state=0).fit(X_train_scaled, y_train)

y_pred = clf_SVC.predict(X_test_scaled)

clfDetails = ['# classes',
              '# samples',
              '# features', 
              'Train score', 
              'Test score', 
              'Accuracy',
              'Precision (weighted)',
              'Precision (micro)',
              'Recall (weighted)',
              'Recall (micro)',
              'F1 (weighted)',
              'F1 (micro)']

clfResults = [len(set(y_train)), # get unique classes from training set
              len(y_train),
              X_train.shape[1],
              str(clf_SVC.score(X_train_scaled, y_train)),
              str(clf_SVC.score(X_test_scaled, y_test)),
              metrics.accuracy_score(y_test, y_pred),
              metrics.precision_score(y_test, y_pred, average='weighted'),
              metrics.precision_score(y_test, y_pred, average='micro'),
              metrics.recall_score(y_test, y_pred, average='weighted'),
              metrics.recall_score(y_test, y_pred, average='micro'),
              metrics.f1_score(y_test, y_pred, average='weighted'),
              metrics.f1_score(y_test, y_pred, average='micro')]

clfResults_df = pd.DataFrame(list(zip(clfDetails, clfResults)), columns=['Name', 'Value'])

clfParams = clf_SVC.get_params()
clfParams = pd.DataFrame.from_dict(clfParams, orient='index')

# %%
# TODO: finish up the feature selection process for SVM.
# NOTE: sort of works. labelling xticks not matching with labels: different data shape.
# NOTE: fortunately feature SVM coefficients/weights are not possible with RBF kernel anyway. Linear kernel only. 

'''cv = CountVectorizer()
cv.fit(X)
feature_names = np.array(cv.get_feature_names())
top_features = 8

coef = clf_SVC.coef_.ravel()
top_positive_coefficients = np.argsort(coef)[-top_features:]
top_negative_coefficients = np.argsort(coef)[:top_features]
top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]

# create plot
featCoefPlot = plt.figure(figsize=(16, 10))
featCoefPlot = plt.ylabel('Coefficient weight')
featCoefPlot = plt.xlabel('Feature')
featCoefPlot = plt.title("SVM feature weights")

featCoefPlot = sns.barplot(x=np.arange(2 * top_features), y=coef[top_coefficients], color='blue')
    
for p in featCoefPlot.patches:
    featCoefPlot.annotate(format(p.get_height(), '.1f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')

loc, labels = plt.xticks()
featCoefPlot.set_xticklabels(feature_names[top_coefficients], rotation=45, ha='right')
#plt.xticks(ticks=[], labels=[feature_names[top_coefficients]], rotation=45, ha='right')
'''
# %%
# create confusion matrix array

cm = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred, labels=uniqueLabel, normalize='true'))
# %%
# plot confusion matrix in heatmap

plt.figure(figsize=(16, 14))
confMatrixFig = sns.heatmap(cm, 
                    annot=True,
                    xticklabels=uniqueLabel,
                    yticklabels=uniqueLabel,
                    cmap='Blues'
                    )
loc, labels = plt.xticks()
confMatrixFig.set_xticklabels(labels, rotation=45, ha='right')

# %%
# export everything

# get time of export
TimeNow = datetime.now() 
TimeNow = TimeNow.strftime('%m-%d-%H-%M')

# export feature importance figure
#fip = featImpPlot.get_figure()
#fip.savefig('D:\Dropbox\Honours\Peter_Woodfordia\Output\SVM\SVM_CLF_FIP_{}.png'.format(TimeNow))

# export feature importance table
#feature_imp_export = feature_imp.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\SVM\SVM_CLF_FI_{}.csv'.format(TimeNow))

# export confusion matrix figure
fig = confMatrixFig.get_figure()
fig.savefig('D:\Dropbox\Honours\Peter_Woodfordia\Output\SVM\SVM_CLF_CM_{}.png'.format(TimeNow))

# export confusion matrix
cm.columns = uniqueLabel # rename matrix columns
cm['Class'] = uniqueLabel # rename matrix rows
cfm = cm.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\SVM\SVM_CLF_CM_{}.csv'.format(TimeNow))

# export classifier results
clfResultOut = clfResults_df.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\SVM\SVM_CLF_Metrics_{}.csv'.format(TimeNow))
clfParams = clfParams.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\SVM\SVM_CLF_Params_{}.csv'.format(TimeNow))

# export features used in classification
clfFeaturesOut = featureList.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\SVM\SVM_CLF_Features_{}.csv'.format(TimeNow))

# export classes and sample size for each
clfClassFreqTrainOut = classFreqTrain.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\SVM\SVM_CLF_ClassFreqTrain_{}.csv'.format(TimeNow))
clfClassFreqTestOut = classFreqTest.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\SVM\SVM_CLF_ClassFreqTest_{}.csv'.format(TimeNow))

# %%
