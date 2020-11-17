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

# %%
# import data
objectFeatures = pd.read_csv('D:\ProgrammingProjects\honours\Data\FeatureStats_BCD.csv')
# %%
# rename columns
objectFeatures.columns = ['ROI',
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
# %%
# split data into train and test sets
####### drop/add object features here #######
X = objectFeatures[objectFeatures.columns[3:]] # x set are the features
y = objectFeatures['class'] # y set is the class labels (in this case species)

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
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# can now drop features that contibute the least