import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


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


# Drop repeated and unnecessary features
data = data.drop(['hour_b', 'total', 'customerAttr_b', 'state'], axis=1)
x_train = x_train.drop(['hour_b', 'total', 'customerAttr_b', 'state'], axis=1)
x_test = x_test.drop(['hour_b', 'total', 'customerAttr_b', 'state'], axis=1)

# Normalize data
x_train = preprocessing.normalize(x_train, norm='l2')
x_test = preprocessing.normalize(x_test, norm='l2')

# Handle Imbalanced data problem
# TODO : Handle Imbalanced data problem


# Decision tree
dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(x_train, y_train)

y_pred_dtc = dtc.predict(x_test)

# Random forest
rfc = RandomForestClassifier()
rfc = rfc.fit(x_train, y_train.values.ravel())

y_pred_rfc = rfc.predict(x_train)

# Neural network
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nn.fit(x_train, y_train.values.ravel())

y_pred_nn = nn.predict(x_test)

# Logistic regression
lr = LogisticRegression()
lr.fit(x_train, y_train.values.ravel())

y_pred_lr = lr.predict(x_test)
