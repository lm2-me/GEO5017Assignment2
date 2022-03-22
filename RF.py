import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm



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

def rf_Plot_max_depth(x_train, x_test, y_train, y_test):
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
        rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        #keep getting NaN for roc_auc
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    line1, = plt.plot(max_depths, train_results, 'b', label ="Train AUC")
    line2, = plt.plot(max_depths, test_results, 'r', label ="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.show()

def rf_Plot_n_estiamators(x_train, x_test, y_train, y_test):
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    train_results = []
    test_results = []
    for estimator in n_estimators:
        rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
    line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.show()