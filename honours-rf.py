#%%
import random
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from datetime import datetime, date, time

# %%
# import data
objectFeatures = pd.read_csv('D:\Dropbox\Honours\Peter_Woodfordia\Data\ALLROI_FS_Merged.csv')

# %%
# define which column caintains the class labels

classLabel = "Genus"

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
classSubset = ['Water',
               'Gravel',
               'Structure']

AllClasses = objectsLabelled[classLabel]
AllClasses = list(AllClasses.unique())
AllClasses.sort()

# this drops classes not in the list above by renaming them to "Non-target"
# this is because it is still useful that they are labelled to reduce comission error
objectsLabelled.loc[objectsLabelled[classLabel].isin(classSubset), ['Species', 'Genus', 'Family']] = "Non-target"
objectsLabelled = objectsLabelled.sort_values(by=[classLabel])

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
# search for best RF hyperparameters (https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# %%

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
# Fit the random search model
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=3,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)
rf_random.fit(X_train_scaled, y_train)

# %%
rf_random.best_params_

# %%
# create and fit model
nEstimators = 400
minSamplesSplit = 5
minSamplesLeaf = 1
maxFeatures = 'sqrt'
maxDepth = 100
bootStrp = True

clf_RF = RandomForestClassifier(n_estimators=nEstimators,
                                min_samples_split=minSamplesSplit,
                                min_samples_leaf=minSamplesLeaf,
                                max_features=maxFeatures,
                                max_depth=maxDepth,
                                bootstrap=bootStrp) # create classifier

clf_RF.fit(X_train_scaled,y_train) # fit classifier
y_pred = clf_RF.predict(X_test_scaled) # predict class using test set
 # %%
 # TODO: Apply classifier to all unclassified objects and export to GIS
 
 predicted = clf_RF.predict(unclassified)
 

# %%
# create metrics
clfDetails = ['# classes',
              '# samples',
              '# features' 
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
              str(clf_RF.score(X_train_scaled, y_train)),
              str(clf_RF.score(X_test_scaled, y_test)),
              metrics.accuracy_score(y_test, y_pred),
              metrics.precision_score(y_test, y_pred, average='weighted'),
              metrics.precision_score(y_test, y_pred, average='micro'),
              metrics.recall_score(y_test, y_pred, average='weighted'),
              metrics.recall_score(y_test, y_pred, average='micro'),
              metrics.f1_score(y_test, y_pred, average='weighted'),
              metrics.f1_score(y_test, y_pred, average='micro')]

clfResults_df = pd.DataFrame(list(zip(clfDetails, clfResults)), columns=['Name', 'Value'])

clfParams = clf_RF.get_params()
clfParams = pd.DataFrame.from_dict(clfParams, orient='index')

# %%
# feature importance
featureNames = list(X) # list of feature names 
# new dataframe with feature importances
feature_imp = pd.Series(clf_RF.feature_importances_, index=featureNames).sort_values(ascending=False)

# %%
# plot importances
featImpPlot = plt.figure(figsize=(16, 10))
featImpPlot = plt.ylabel('Feature')
featImpPlot = plt.xlabel('Feature Importance Score')
featImpPlot = plt.title("Feature importance for all classes")

featImpPlot = sns.barplot(x=feature_imp, y=feature_imp.index, color='blue')

for p in featImpPlot.patches:
    width = p.get_width()
    plt.text(0.002+p.get_width(), p.get_y()+0.55*p.get_height(),
             '{:1.2f}'.format(width),
             ha='center', va='center')

# can now drop features that contibute the least

# %%
'''# Generate confusion matrix with built-in sklearn
fig, ax = plt.subplots(figsize=(10, 10))
matrix = metrics.plot_confusion_matrix(clf_RF, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 ax=ax)
plt.title('Confusion matrix for our classifier')
plt.xticks(rotation=45)
plt.show(matrix)'''
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
fip = featImpPlot.get_figure()
fip.savefig('D:\Dropbox\Honours\Peter_Woodfordia\Output\RF\RF_CLF_FIP_{}.png'.format(TimeNow))

# export feature importance table
feature_imp_export = feature_imp.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\RF\RF_CLF_FI_{}.csv'.format(TimeNow))

# export confusion matrix figure
fig = confMatrixFig.get_figure()
fig.savefig('D:\Dropbox\Honours\Peter_Woodfordia\Output\RF\RF_CLF_CM_{}.png'.format(TimeNow))

# export confusion matrix
cm.columns = uniqueLabel # rename matrix columns
cm['Class'] = uniqueLabel # rename matrix rows
cfm = cm.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\RF\RF_CLF_CM_{}.csv'.format(TimeNow))

# export classifier results
clfResultOut = clfResults_df.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\RF\RF_CLF_Metrics_{}.csv'.format(TimeNow))
clfParams = clfParams.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\RF\RF_CLF_Params_{}.csv'.format(TimeNow))
# export features used in classification
clfFeaturesOut = featureList.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\RF\RF_CLF_Features_{}.csv'.format(TimeNow))

# export classes and sample size for each
clfClassFreqTrainOut = classFreqTrain.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\RF\RF_CLF_ClassFreqTrain_{}.csv'.format(TimeNow))
clfClassFreqTestOut = classFreqTest.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Output\RF\RF_CLF_ClassFreqTest_{}.csv'.format(TimeNow))

# %%
