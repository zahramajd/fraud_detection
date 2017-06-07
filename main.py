import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def get_result(predicted):
    print "F1_Score: " + str(f1_score(y_validation, predicted, average='macro'))
    print "accuracy: " + str(accuracy_score(y_validation, predicted))
    print "AUC: " + str(roc_auc_score(y_validation, predicted))
    print "recall: " + str(recall_score(y_validation, predicted))
    return


# Load data
x_train = pd.read_csv("data_fraud/X_train.csv")
y_train = pd.read_csv("data_fraud/Y_train.csv")
x_test = pd.read_csv("data_fraud/X_test.csv")

# Draw correlation matrix
# data = pd.concat([x_train, y_train])
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

# Split data to validation and train
train_percent = 0.66
validate_percent = 0.33
m = len(x_train)
x_train = x_train[:int(train_percent * m)]
x_validation = x_train[int(validate_percent * m):]
y_train = y_train[:int(train_percent * m)]
y_validation = y_train[int(validate_percent * m):]

# Drop repeated and unnecessary features
x_train = x_train.drop(['hour_b', 'total', 'customerAttr_b', 'zip', 'state'], axis=1)
x_test = x_test.drop(['hour_b', 'total', 'customerAttr_b', 'zip', 'state'], axis=1)
x_validation = x_validation.drop(['hour_b', 'total', 'customerAttr_b', 'zip', 'state'], axis=1)

# Normalize data
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)
x_validation = min_max_scaler.fit_transform(x_validation)

# Handle Imbalanced data problem
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_sample(x_train, y_train)

# Decision tree
dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(x_train, y_train)
y_predicted_validation_dtc = dtc.predict(x_validation)
# y_predicted_test_dtc = dtc.predict(x_test)

print "- Decision tree -"
get_result(y_predicted_validation_dtc)

# Random forest
rfc = RandomForestClassifier()
rfc = rfc.fit(x_train, y_train)
y_predicted_validation_rfc = rfc.predict(x_validation)
# y_prediction_test_rfc = rfc.predict(x_train)

print "- Random forest -"
get_result(y_predicted_validation_rfc)

# Neural network
# TODO: debug
# nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# nn.fit(x_train, y_train)
# y_predicted_validation_nn = nn.predict(x_validation)
# y_predicted_test_nn = nn.predict(x_test)
#
# print "- Neural network -"
# print "F1_Score: " + str(f1_score(y_validation, y_predicted_validation_nn, average='macro'))
# print "accuracy: " + str(accuracy_score(y_validation, y_predicted_validation_nn))
# print "AUC: " + str(roc_auc_score(y_validation, y_predicted_validation_nn))
# print "recall: " + str(recall_score(y_validation, y_predicted_validation_nn))

#
# Logistic regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_predicted_validation_lr = lr.predict(x_validation)
# y_prediction_test_lr = lr.predict(x_test)

print "- Logistic regression -"
get_result(y_predicted_validation_lr)

# Ensemble of classifiers

# Voting classifier
vc = VotingClassifier(estimators=[('dt', dtc), ('rf', rfc), ('lr', lr)], voting='soft')
vc.fit(x_train, y_train)
y_predicted_validation_vc = vc.predict(x_validation)
# y_prediction_test_vc = vc.predict(x_test)

print "- Voting -"
get_result(y_predicted_validation_vc)

# AdaBoost classifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
bdt.fit(x_train, y_train)
y_predicted_validation_bdt = vc.predict(x_validation)
# y_prediction_test_bdt = vc.predict(x_test)

print "- AdaBoost -"
get_result(y_predicted_validation_bdt)

# Gradient Boosting classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
gb.fit(x_train, y_train)
y_predicted_validation_gb = gb.predict(x_validation)
# y_prediction_test_gb = vc.predict(x_test)

print "- Gradient Boosting -"
get_result(y_predicted_validation_gb)

# Bagging classifier

# pd.DataFrame({'frud': y_predicted_test_nn}).to_csv('P2_submission.csv', index =False)
