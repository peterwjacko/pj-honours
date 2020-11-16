#1 Importing the librariesimport numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#2 Importing the datasetdataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values.astype(float)
y = dataset.iloc[:,2:3].values.astype(float)
#3 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
#4 Fitting the Support Vector Regression Model to the dataset
# Create your support vector regressor herefrom sklearn.svm import SVR
# # most important SVR parameter is Kernel type. It can be 
# #linear,polynomial or gaussian SVR. We have a non-linear condition 
# #so we can select polynomial or gaussian but here we select RBF(a 
# #gaussian type) kernel.regressor = SVR(kernel='rbf')
regressor.fit(X,y)#5 Predicting a new result
y_pred = regressor.predict(6.5)