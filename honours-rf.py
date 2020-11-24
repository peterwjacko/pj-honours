#%%
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from datetime import datetime, date, time

# %%
# import data
objectFeatures = pd.read_csv('D:\Dropbox\Honours\Peter_Woodfordia\Data\ALLROI_FS_Merged.csv')
# %%
# rename columns
'''objectFeatures.columns = ['ROI',
                        'set',
                        'label',
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
                        'stdGLCM']'''
# %%
# create subset of labelled objects
objectsLabelled = objectFeatures[objectFeatures["Class"].notna()] 

# create subset of unlabelled objects
objectsUnlabelled = objectFeatures[objectFeatures["Class"].isna()] 
# %%
#objectsUnlabelled.columns
# %%
# split data into train and test sets
####### drop/add object features here #######
'''['ROI', 'Height', 'Class', 'Method', 'Family', 'Genus', 'ObjectRef',
    'Asymmetry', 'Border_index', 'Compactness', 'GLCM_Ang_2',
    'GLCM_Dissimilarity', 'GLCM_Entropy', 'GLCM_Homogeneity', 'GLCM_Mean',
    'GLCM_StdDev', 'Max_diff', 'Max_lidarCHM', 'Mean_blue', 'Mean_GLI',
    'Mean_green', 'Mean_GRVI', 'Mean_LidarCHM', 'Mean_red', 'Mean_VARI',
    'Mean_VVI', 'Roundness', 'Shape_index', 'Std_blue', 'Std_GLI',
    'Std_green', 'Std_GRVI', 'Std_lidarCHM', 'Std_red', 'Std_VARI',
    'Std_VVI']'''
X = objectsLabelled[objectsLabelled.columns[7:]] # x set are the features
y = objectsLabelled['Class'] # y set is the class labels (in this case species)

uniqueLabel = list(objectsLabelled.Class.unique()) # get unique labels/classes
uniqueLabel.sort() # sort in alphabetical

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
# create and fit model
clf_RF = RandomForestClassifier(n_estimators=100) # create classifier

clf_RF.fit(X_train,y_train) # fit classifier
y_pred = clf_RF.predict(X_test) # predict class using test set
 # %%
 # TODO: Apply classifier to all unclassified objects and export to GIS
 
 predicted = clf_RF.predict(unclassified)
 

# %%
# check accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# %%
# feature importance
featureNames = list(X) # list of feature names 
# new dataframe with feature importances
feature_imp = pd.Series(clf_RF.feature_importances_, index=featureNames).sort_values(ascending=False)
feature_imp.head()

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
# Generate confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
matrix = metrics.plot_confusion_matrix(clf_RF, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 ax=ax)
plt.title('Confusion matrix for our classifier')
plt.xticks(rotation=45)
plt.show(matrix)
# %%
cm = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred, labels=uniqueLabel, normalize='true'))

# %%
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
fip.savefig('D:\Dropbox\Honours\Peter_Woodfordia\Data\ClassificationOutput\RF\RF_CLF_FIP_{}.png'.format(TimeNow))

# export feature importance table
feature_imp_export = feature_imp.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Data\ClassificationOutput\RF\RF_CLF_FI_{}.csv'.format(TimeNow))

# export confusion matrix figure
fig = confMatrixFig.get_figure()
fig.savefig('D:\Dropbox\Honours\Peter_Woodfordia\Data\ClassificationOutput\RF\RF_CLF_CM_{}.png'.format(TimeNow))

# export confusion matrix
cm.columns = uniqueLabel # rename matrix columns
cm['Class'] = uniqueLabel # rename matrix rows
cfm = cm.to_csv('D:\Dropbox\Honours\Peter_Woodfordia\Data\ClassificationOutput\RF\RF_CLF_CM_{}.csv'.format(TimeNow))


# %%
