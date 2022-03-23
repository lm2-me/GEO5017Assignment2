import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

#import os
# cwd = os.getcwd()
# print(cwd)


# np.savetxt('features_norm.csv', normalized_object_features, delimiter=',')
dataload = np.loadtxt('features_norm_new.csv', delimiter=',')

# X, y = np.array(dataload[:,1:6]) , dataload[:,0]

features_only = dataload

y_true = np.loadtxt('y_true.csv', dtype=str)


X = features_only
y = y_true

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


classifier = svm.SVC(kernel='linear', C=1000)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


test_size_neg = []
test_size_record = []
AO = []

test_size = 0
while test_size < 0.99:
    test_size += 0.01
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    classifier = svm.SVC(kernel='rbf', C=1,decision_function_shape='ovr')
    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    test_size_record.append(test_size)
    AO.append(accuracy)
    test_size_neg.append(1-test_size)
    # print(test_size)


plt.scatter(test_size_neg, AO)
plt.xlabel("Train data size")
plt.ylabel("Overall Accuracy")
plt.show()
