import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import Evaluation as ev

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
        clf = RandomForestClassifier(n_estimators=200, criterion='gini', max_features='auto', bootstrap=True, max_samples=None)
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
    #log_fit = np.polyfit(np.log(test_size_neg), AO, 2)
    #y_line = log_fit[0] * np.log(test_size_neg) ** 2 + log_fit[1] * np.log(test_size_neg) + log_fit[2]
    
    plt.scatter(test_size_neg, AO)
    #plt.plot(test_size_neg, y_line, 'r')
    plt.xlabel('Percentage Training Data')
    plt.ylabel('Overall Accuracy')
    plt.show()

    return y_pred, y_test

def rf_Plot_max_depth(x_train, x_test, y_train, y_test):
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
        rf = RandomForestClassifier(n_estimators=200, criterion='gini', max_features='auto', bootstrap=True, max_samples=None, max_depth=max_depth)
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        # keep getting NaN for roc_auc
        # roc_auc = auc(false_positive_rate, true_positive_rate)
        rf_oa_train = ev.overallAccuracy(y_train, train_pred)
        train_results.append(rf_oa_train)
        test_pred = rf.predict(x_test)
        # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, test_pred)
        # roc_auc = auc(false_positive_rate, true_positive_rate)
        rf_oa_test = ev.overallAccuracy(y_test, test_pred)
        test_results.append(rf_oa_test)

    line1, = plt.plot(max_depths, train_results, 'b', label ="Train Overall accuracy")
    line2, = plt.plot(max_depths, test_results, 'r', label ="Test Overall accuracy")
   
    plt.scatter(max_depths, train_results)
    plt.scatter(max_depths, test_results)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Overall accuracy')
    plt.xlabel('Tree depth')
    plt.show()

def rf_Plot_n_estiamators(x_train, x_test, y_train, y_test):
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    train_results = []
    test_results = []
    for estimator in n_estimators:
        rf = RandomForestClassifier(n_estimators=estimator)
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        rf_oa_train = ev.overallAccuracy(y_train, train_pred)
        train_results.append(rf_oa_train)
        test_pred = rf.predict(x_test)
        rf_oa_test = ev.overallAccuracy(y_test, test_pred)
        test_results.append(rf_oa_test)

    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(n_estimators, train_results, 'b', label="Train Overall accuracy")
    line2, = plt.plot(n_estimators, test_results, 'r', label="Test Overall accuracy")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Overall accuracy')
    plt.xlabel('n_estimators')
    plt.show()

    