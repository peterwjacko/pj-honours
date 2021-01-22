#%%
import random
import numpy as np 
import pandas as pd
 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from datetime import datetime, date, time
from itertools import chain

# %%
# set feature groupings here

featuresDict = {'featuresAll': ['Asymmetry', 'Border_index', 'Compactness', 'GLCM_Ang_2',
                            'GLCM_Dissimilarity', 'GLCM_Entropy', 'GLCM_Homogeneity', 'GLCM_Mean',
                            'GLCM_StdDev', 'Max_diff', 'Max_lidarCHM', 'Mean_blue', 'Mean_GLI',
                            'Mean_green', 'Mean_GRVI', 'Mean_LidarCHM', 'Mean_red', 'Mean_VARI',
                            'Mean_VVI', 'Roundness', 'Shape_index', 'Std_blue', 'Std_GLI',
                            'Std_green', 'Std_GRVI', 'Std_lidarCHM', 'Std_red', 'Std_VARI',
                            'Std_VVI'],
            'featuresSpectral': ['Max_diff',
                                 'Mean_blue',
                                 'Mean_green',
                                 'Mean_red',
                                 'Std_blue',
                                 'Std_green',
                                 'Std_red'],
            'featuresCHM': ['Max_lidarCHM',
                            'Mean_LidarCHM',
                            'Std_lidarCHM'],
            'featuresVegIndex': ['Mean_GLI',
                                 'Mean_GRVI',
                                 'Mean_VARI',
                                 'Mean_VVI',
                                 'Std_GLI',
                                 'Std_GRVI',
                                 'Std_VARI',
                                 'Std_VVI'],
            'featuresTextural': ['GLCM_Ang_2',
                                 'GLCM_Dissimilarity',
                                 'GLCM_Entropy',
                                 'GLCM_Homogeneity',
                                 'GLCM_Mean',
                                 'GLCM_StdDev'],
            'featuresGeom': ['Asymmetry',
                             'Border_index',
                             'Compactness'],
            'featuresRandom': random.sample, # TODO: this needs to be set up. When calling this key:value pair, add the arguments onto the script (featuresAll, 7)
            'featuresCustomSubset': []}
#%%
# feature combinations
def featuresActive(featuresCombo, featuresDict):
    featuresActiveList = []
    for featSet in featuresCombo:
        featuresActiveList.append(featuresDict.get(featSet))
    featuresActiveList = list(chain.from_iterable(featuresActiveList))
    return featuresActiveList              
# %%
# relabels classes to non-target
def activeLabels(classSubset, objectsLabelled, classLabel):
    classSubset = classSubset
    objectsLabelled = objectsLabelled
    objectsLabelled.loc[~objectsLabelled[classLabel].isin(classSubset), ['Species', 'Genus', 'Family']] = "Non-target"
    objectsLabelled = objectsLabelled.sort_values(by=[classLabel])
    return objectsLabelled
# %%
# split data into train and test sets
def splitData(featuresActiveList, classLabel, objectsLabelled, testSize):
    X = objectsLabelled[featuresActiveList] # x set are the features
    y = objectsLabelled[classLabel] # y set is the class labels (in this case species)
    uniqueLabel = objectsLabelled[classLabel]
    uniqueLabel = list(uniqueLabel.unique()) # get unique labels/classes
    uniqueLabel.sort() # sort in alphabetical
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=1, stratify=y)
    return uniqueLabel, X_train, X_test, y_train, y_test, X
# %%
# create metrics about the data
def dataMetrics(y_train, y_test, X_train):
    classFreqTrain = y_train.value_counts()
    classFreqTest = y_test.value_counts()
    featureList = list(X_train.columns)
    featureList = pd.DataFrame(featureList)
    return classFreqTrain, classFreqTest, featureList
# %%
# scale/transform data here
def dataScaling (transformType, X_train, X_test):
    if transformType == "MinMax":
        scaler = MinMaxScaler()
    elif transformType == "Standard":
        scaler = StandardScaler()
    elif transformType == "Normalize":
        scaler = Normalizer()
    else:
        print("No data transformation selected. Using original values.")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return(X_train_scaled, X_test_scaled)
# %%
# create RF hyperparamter values grid
def hyperparameterGridRF():
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # Number of trees in random forest
    max_features = ['auto', 'sqrt'] # Number of features to consider at every split
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # Maximum number of levels in tree
    max_depth.append(None)
    min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
    min_samples_leaf = [1, 2, 4] # Minimum number of samples required at each leaf node
    bootstrap = [True, False] # Method of selecting samples for training each tree
    # Create the random grid
    paramGrid_rf = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    return paramGrid_rf
# %%
# test RF hyperparamter values grid
def testHyperparamsRF(paramGrid_rf, X_train_scaled, X_test_scaled, y_train, y_test):
    rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                param_distributions=paramGrid_rf,
                                n_iter=100,
                                cv=3,
                                verbose=2,
                                random_state=42,
                                n_jobs=-1)
    rf_random.fit(X_train_scaled, y_train)
    bestParams_rf = rf_random.best_params_
    print(bestParams_rf)
    print("Train score:" + str(rf_random.score(X_train_scaled, y_train)))
    print("Test score:" + str(rf_random.score(X_test_scaled, y_test)))
    return bestParams_rf
# %%
# create and fit model
def applyRF(hyperParameters, X_train_scaled, X_test_scaled, y_train, y_test):
    clf_RF = RandomForestClassifier() # create classifier
    clf_RF.set_params(**hyperParameters)
    clf_RF.fit(X_train_scaled, y_train) # fit classifier
    y_pred = clf_RF.predict(X_test_scaled) # predict class using test set
    trainScore = clf_RF.score(X_train_scaled, y_train)
    testScore = clf_RF.score(X_test_scaled, y_test)
    return y_pred, clf_RF, trainScore, testScore
# %%
# feature importance
def getFI(clf_RF, X):
    featureNames = list(X) # list of feature names
    feature_imp = pd.Series(clf_RF.feature_importances_, index=featureNames).sort_values(ascending=False)
    return feature_imp
# %%
# plot importances
def FI_plot(feature_imp):
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
    return featImpPlot    
# %%
# create SVM hyperparameter values grid
def hyperparameterGridSVM():
    paramGrid_svm = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 150, 200, 300, 400, 600, 800, 1000],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    return paramGrid_svm 
# %%
# test SVM hyperparamter values grid
def testHyperparamsSVM(paramGrid_svm, X_train_scaled, X_test_scaled, y_train, y_test):
    svm_random = RandomizedSearchCV(estimator=svm.SVC(),
                                param_distributions=paramGrid_svm,
                                n_iter=100,
                                cv=3,
                                verbose=2,
                                random_state=42,
                                n_jobs=-1)

    svm_random.fit(X_train_scaled, y_train)
    bestParams_svm = svm_random.best_params_
    print(bestParams_svm)
    print("Train score:" + str(svm_random.score(X_train_scaled, y_train)))
    print("Test score:" + str(svm_random.score(X_test_scaled, y_test)))
    return bestParams_svm
# %%
# fit model with optimal parameters and test
def applySVM(hyperParameters, X_train_scaled, X_test_scaled, y_train, y_test):
    clf_SVC = svm.SVC() # create classifier
    clf_SVC.set_params(**hyperParameters)
    clf_SVC.fit(X_train_scaled, y_train) # fit classifier
    y_pred = clf_SVC.predict(X_test_scaled) # predict class using test set
    trainScore = clf_SVC.score(X_train_scaled, y_train)
    testScore = clf_SVC.score(X_test_scaled, y_test)
    return y_pred, clf_SVC, trainScore, testScore
# %%
# create metrics
def modelInfo(clf, y_test, y_train, y_pred, trainScore, testScore, X_train_scaled):
    clfMetrics = {'# classes': len(set(y_train)), 
                  '# samples': len(y_train),
                  '# features': X_train_scaled.shape[1],
                  'Train score': str(trainScore),
                  'Test score': str(testScore),
                  'Accuracy': metrics.accuracy_score(y_test, y_pred),
                  'Precision (weighted)': metrics.precision_score(y_test, y_pred, average='weighted'),
                  'Precision (micro)': metrics.precision_score(y_test, y_pred, average='micro'),
                  'Recall (weighted)': metrics.recall_score(y_test, y_pred, average='weighted'),
                  'Recall (micro)': metrics.recall_score(y_test, y_pred, average='micro'),
                  'F1 (weighted)': metrics.f1_score(y_test, y_pred, average='weighted'),
                  'F1 (micro)': metrics.f1_score(y_test, y_pred, average='micro')}
    clfMetrics_df = pd.DataFrame.from_dict(clfMetrics, orient='index')
    clfParams_df = pd.DataFrame.from_dict(clf.get_params(), orient='index')
    return clfMetrics_df, clfParams_df    
# %%
# create confusion matrix array
def confMatrix(y_test, y_pred, uniqueLabel):
    confMat = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred, labels=uniqueLabel, normalize='true'))
    #confMat.columns = uniqueLabel
    #confMat['Class'] = uniqueLabel # TODO: make sue this is in the correct order (matrix not missmatching)
    return confMat
# %%
# plot confusion matrix
def confMatrixPlot(confMat, uniqueLabel):
    plt.figure(figsize=(16, 14))
    confMatrixFig = sns.heatmap(confMat, 
                        annot=True,
                        xticklabels=uniqueLabel,
                        yticklabels=uniqueLabel,
                        cmap='Blues'
                        )
    loc, labels = plt.xticks()
    confMatrixFig.set_xticklabels(labels, rotation=45, ha='right')
    return confMatrixFig
# %%
# export everything
# create a directory that has ~/output/SVM or ~/output/RF
def exportAll(activeModel, confMat, 
              confMatrixFig, clfMetrics_df, 
              clfParams_df, featureList, 
              classFreqTrain, classFreqTest, 
              featImpPlot, feature_imp):
    TimeNow = datetime.now() 
    TimeNow = TimeNow.strftime('%m-%d-%H-%M')
    outputPath = str(f'D:\Dropbox\Honours\Peter_Woodfordia\Output\{activeModel}\{activeModel}_')
    fig = confMatrixFig.get_figure()
    fig.savefig(f'{outputPath}CLF_CM_{TimeNow}.png')
    cfm = confMat.to_csv(f'{outputPath}CLF_CM_{TimeNow}.csv')
    clfResultOut = clfMetrics_df.to_csv(f'{outputPath}CLF_Metrics_{TimeNow}.csv')
    clfParams = clfParams_df.to_csv(f'{outputPath}CLF_Params_{TimeNow}.csv')
    clfFeaturesOut = featureList.to_csv(f'{outputPath}CLF_Features_{TimeNow}.csv')
    clfClassFreqTrainOut = classFreqTrain.to_csv(f'{outputPath}CLF_ClassFreqTrain_{TimeNow}.csv')
    clfClassFreqTestOut = classFreqTest.to_csv(f'{outputPath}CLF_ClassFreqTest_{TimeNow}.csv')
    if feature_imp is not 0:
        fip = featImpPlot.get_figure()
        fip.savefig(f'{outputPath}CLF_FIP_{TimeNow}.png')
        feature_imp_export = feature_imp.to_csv(f'{outputPath}_CLF_FI_{TimeNow}.csv')
    elif feature_imp == 0:
        print("SVM: No feature importance generated")
# %%
# put it all together
def runClassification(classLabel, objectsLabelled, 
                      featuresCombo, classSubset, 
                      testSize, transformType, 
                      activeModel, runHyperparamTest, 
                      customParams_rf, customParams_svm, featuresDict):
    featuresActiveList = featuresActive(featuresCombo, featuresDict)
    objectsLabelled = activeLabels(classSubset, objectsLabelled, classLabel)
    uniqueLabel, X_train, X_test, y_train, y_test, X = splitData(featuresActiveList, classLabel, objectsLabelled, testSize)
    classFreqTrain, classFreqTest, featureList = dataMetrics(y_train, y_test, X_train)
    X_train_scaled, X_test_scaled = dataScaling(transformType, X_train, X_test)
    # classification
    if activeModel == 'RF':
        if runHyperparamTest == 'on':
            paramGrid_rf = hyperparameterGridRF()
            bestParams_rf = testHyperparamsRF(paramGrid_rf, X_train_scaled, X_test_scaled, y_train, y_test)
            hyperParameters = bestParams_rf
            y_pred, clf, trainScore, testScore = applyRF(hyperParameters, X_train_scaled, X_test_scaled, y_train, y_test)
        elif runHyperparamTest == 'off':
            hyperParameters = customParams_rf
            y_pred, clf, trainScore, testScore = applyRF(hyperParameters, X_train_scaled, X_test_scaled, y_train, y_test)
    elif activeModel == 'SVM':
        if runHyperparamTest == 'on':
            paramGrid_svm = hyperparameterGridSVM()
            bestParams_svm = testHyperparamsSVM(paramGrid_svm, X_train_scaled, X_test_scaled, y_train, y_test)
            hyperParameters = bestParams_svm
            y_pred, clf, trainScore, testScore = applySVM(hyperParameters, X_train_scaled, X_test_scaled, y_train, y_test)
        elif runHyperparamTest == 'off':
            hyperParameters = customParams_svm
            y_pred, clf, trainScore, testScore = applySVM(hyperParameters, X_train_scaled, X_test_scaled, y_train, y_test)
    # create outputs
    clfMetrics_df, clfParams_df = modelInfo(clf, y_test, y_train, y_pred, trainScore, testScore, X_train_scaled)
    confMat = confMatrix(y_test, y_pred, uniqueLabel)
    confMatrixFig = confMatrixPlot(confMat, uniqueLabel)
    if activeModel == 'RF':
        feature_imp = getFI(clf, X)
        featImpPlot = FI_plot(feature_imp)
    elif activeModel == 'SVM':
        feature_imp = 0
        featImpPlot = 0
    # export
    exportAll(activeModel, confMat, confMatrixFig, 
              clfMetrics_df, clfParams_df, featureList, 
              classFreqTrain, classFreqTest, featImpPlot, 
              feature_imp)

# %%
## variables
# data table
# TODO: write function for importing
objectFeatures = pd.read_csv('D:\Dropbox\Honours\Peter_Woodfordia\Data\ALLROI_FS_Merged.csv')
# column that contains the class label
classLabel = 'Genus'
# subsets labelled and unlabelled data
objectsLabelled = objectFeatures[objectFeatures[classLabel].notna()]
objectsUnlabelled = objectFeatures[objectFeatures[classLabel].isna()] 
# list of feature sets to be used
featuresCombo = ['featuresAll']
#'featuresSpectral',
                 #'featuresCHM',
                 #featuresVegIndex,
                 #featuresTextural,
                 #'featuresGeom']
                 #featuresRandom]
# list which classes to keep 
classSubset = ['Lantana']

# % of data for testing
testSize = float(0.33)
# data transformation type ("MinMax", "Standard", "Normalize", None)
transformType = 'MinMax'
# which model to use
activeModel = 'RF' 
# run random grid test on hyperparams?
runHyperparamTest = 'on'
# enter hyperparameters here:
customParams_rf = {'n_estimators': 400,
               'max_features': 'sqrt',
               'max_depth': 1,
               'min_samples_split': 5,
               'min_samples_leaf': 100,
               'bootstrap': True}

customParams_svm = {'C': 200,
                    'gamma': 0.1,
                    'kernel': 'rbf'}
 
# %%
runClassification(classLabel, 
                  objectsLabelled, 
                  featuresCombo, 
                  classSubset, 
                  testSize, 
                  transformType, 
                  activeModel, 
                  runHyperparamTest, 
                  customParams_rf, 
                  customParams_svm,
                  featuresDict)

# %%

