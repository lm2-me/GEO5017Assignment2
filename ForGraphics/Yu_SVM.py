import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import operator

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
    classifier.fit(X_train, y_train)

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



def graph_svm():
    dataload = np.loadtxt('features_norm_new.csv', delimiter=',')
    dataload = dataload[:, [6, 2, 0]]
    list = np.arange(500).tolist()
    list_correct = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=17)
    X_list_train, X_list_test, y_list_train, y_list_test = train_test_split(list, y, test_size=0.4,
                                                                            random_state=17)

    classifier = svm.SVC(kernel='rbf', C=1,decision_function_shape='ovr')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    for i, pred in enumerate(y_pred):
        if pred == y_test[i]:
            list_correct.append( int(i))

    list_correct = np.sort(list_correct)
    # correct_data = X_test[int(list_correct)]
    # correct_data = operator.itemgetter(*int(list_correct))(X_test)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim(0.4, 1)
    ax.set_ylim(0, 0.8)
    ax.set_zlim(0, 0.8)
    for i in range(len(dataload)):
        if i in list_correct:
            x_correct = dataload[i, 0]
            y_correct = dataload[i, 1]
            z_correct = dataload[i, 2]
            ax.scatter(x_correct, y_correct, z_correct, marker='o', alpha=0.5, c='grey')
        if i not in list_correct:
            x_error = dataload[i, 0]
            y_error = dataload[i, 1]
            z_error = dataload[i, 2]
            ax.scatter(x_error, y_error, z_error, marker='^', alpha=1, c='red')

    ax.set_xlabel('Squareness')
    ax.set_ylabel('Average Height')
    ax.set_zlabel('Height')
    plt.show()
