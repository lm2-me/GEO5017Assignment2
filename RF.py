import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
import math
import scipy
import Evaluation as ev




def splitdata(features_only, y_true):
    X = features_only
    y = y_true

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    return X_train, X_test, y_train, y_test



def randomforest(X_train, X_test, y_train, y_test):
    #max_depth and min_samples_leaf do not need to be set because our data set is small

    #create classifier object
    clf = RandomForestClassifier(n_estimators=200, criterion='gini', max_features='auto', bootstrap=True, max_samples=None, max_depth=4)
    #train classifier object
    clf.fit(X_train, y_train)
    #test classifier
    y_pred = clf.predict(X_test)

    return y_pred, y_test

def learningcurve(features_only, y_true):
    X = features_only
    y = y_true
       
    test_size_neg = []
    test_size_record = []
    AO = []

    test_size = 0
    while test_size < 0.99:
        test_size += 0.01
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        #create classifier object
        clf = RandomForestClassifier(n_estimators=200, criterion='gini', max_features='auto', bootstrap=True, max_samples=None, max_depth=4)
        #train classifier object
        clf.fit(X_train, y_train)
        #test classifier
        y_pred = clf.predict(X_test)
        accuracy = ev.overallAccuracy(y_test, y_pred)

        
        test_size_record.append(test_size)
        AO.append(accuracy)

    for size in test_size_record:
        test_size_neg.append(1-size)

    #X
    test_size_neg = np.array(test_size_neg)
    #Y
    AO = np.array(AO)

    #second order log fit line to data
    log_fit = np.polyfit(np.log(test_size_neg), AO, 2)
    y_line = log_fit[0] * np.log(test_size_neg) ** 2 + log_fit[1] * np.log(test_size_neg) + log_fit[2]

    plt.scatter(test_size_neg, AO)
    plt.plot(test_size_neg, y_line, 'r')
    plt.xlabel('Percentage Training Data')
    plt.ylabel('Overall Accuracy')
    plt.show()