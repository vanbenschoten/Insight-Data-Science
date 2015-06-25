#Import necesary Python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

#Read company data into pandas dataframe
#NOTE: PostgreSQL database work cannot be shown, so we're assuming the pertinent data is stored in a csv file
data = pd.read_csv('data.csv')

#Select a subset of the listings and drop rows with null values in certain columns
data = data.ix[(data.status == 'condition_1') | (data.status == 'condition_2')]
data = data.dropna(subset=['column_1','column_2'])

#'Define a new metric
data['new_metric'] = data.column_x/data.column_y

#Examine dataframe and remove features with low non-null counts
data.info()
data = data.drop(['various_features'],axis=1)

#Convert specific fields to Boolean
for i in ('various_features'):
    data[i] = data[i].notnull()

#Get necessary dummy variables for classification features
df = pd.concat([df, pd.get_dummies(df.column_z,prefix='feature')],axis=1)

#Create non-skewed dataset
skew = pd.read_pickle('company_data.pickle')
good = skew.ix[skew.outcome == 1]
bad = skew.ix[skew.outcome == 0]
rows = random.sample(good.index,num)
good_sample = good.ix[rows]
combined = pd.concat([bad,good_sample])


##Examples of basic data exploration in iPython

#Number of employees
emp = combined.number_of_employees
plt.figure()
emp.plot(kind='hist')

#Looking at text length of descriptions
des = combined.responsibilities_length
plt.figure()
des.plot(kind='hist')


##Data cleaning/converting
#Applying log (+1) transformations to certain columns
for i in ("various features"):
    combined[i+'_log'] = np.log10(combined[i]+1)


##Centering and normalization of features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

y = combined.outcome.values

x = combined.drop('another_column',axis=1).as_matrix().astype(np.float)

X = scaler.fit_transform(x)

#Performing leave-one-out cross validation on a logistic regression model (can replace with other classifiers)
from sklearn.linear_model import LogisticRegression
dummy = np.zeros(25)
from sklearn.cross_validation import LeaveOneOut
loo = LeaveOneOut(num)
model = LogisticRegression()
accuracy = []
for train_index, test_index in loo:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = model.fit(X_train, y_train)
    accuracy.append(model.score(X_test,y_test))
    dummy2 = model.coef_.flatten()
    q = np.vstack([dummy,dummy2])
    dummy = q


#Obtaining mean/standard deviation of features for each training/validation set
features_mean = np.mean(dummy[1:,:], axis=0)
features_std_dev = np.std(dummy[1:,:], axis=0)

#Calculating a confusion matrix
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y,prediction)
