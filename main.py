import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Load data
x_train = pd.read_csv("data_fraud/X_train.csv")
y_train = pd.read_csv("data_fraud/Y_train.csv")
x_test = pd.read_csv("data_fraud/X_test.csv")

data = pd.concat([x_train, y_train])

# Draw correlation matrix
# corr = data.corr()
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(12, 9))
#
# # Draw the heatmap using seaborns
# sns.heatmap(corr, vmax=.8, square=True)
# plt.show()


# Convert to dummies
m = len(x_train)
n = len(x_test)

x = pd.concat([x_train, x_test])

states = pd.get_dummies(x['state'])
x = pd.concat([x, states], axis=1)

x_train = x[0:m]
x_test = x[m:m + n]

# # Convert to dummies
# x=pd.get_dummies(x_train['state'])
# x_train=pd.concat([x_train, x],axis=1)
#
# # x=pd.get_dummies(x_test['state'])
# x_test=pd.concat([x_test, x], axis=1)
#
# x_test.fillna('0')

# Split data to validation and train
train_percent = 0.66
validate_percent = 0.33
m = len(x_train)
x_train = x_train[:int(train_percent * m)]
x_validation = x_train[int(validate_percent * m):]
y_train = y_train[:int(train_percent * m)]
y_validation = y_train[int(validate_percent * m):]

# Drop repeated and unnecessary features
data = data.drop(['hour_b', 'total', 'customerAttr_b', 'zip', 'state'], axis=1)
x_train = x_train.drop(['hour_b', 'total', 'customerAttr_b', 'zip', 'state'], axis=1)
x_test = x_test.drop(['hour_b', 'total', 'customerAttr_b', 'zip', 'state'], axis=1)
x_validation = x_validation.drop(['hour_b', 'total', 'customerAttr_b', 'zip', 'state'], axis=1)



# Normalize data
# x_train = preprocessing.normalize(x_train, norm='l2')
#x_test = preprocessing.normalize(x_test, norm='l2')
#x_validation = preprocessing.normalize(x_validation, norm='l2')


# Handle Imbalanced data problem
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_sample(x_train, y_train)

# # Decision tree
dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(x_train, y_train)

y_pred_dtc = dtc.predict(x_test)


# # Random forest
# rfc = RandomForestClassifier()
# rfc = rfc.fit(x_train, y_train)
#
# y_pred_rfc = rfc.predict(x_train)
#
# # Neural network
# nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# nn.fit(x_train, y_train)
#
# y_pred_nn = nn.predict(x_test)
#
# # Logistic regression
# lr = LogisticRegression()
# lr.fit(x_train, y_train)
#
# y_pred_lr = lr.predict(x_test)
