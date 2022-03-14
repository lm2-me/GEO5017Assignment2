import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm



def splitdata(features_only):
    y_true = np.loadtxt('y_true.csv', dtype=str)
    X = features_only
    y = y_true

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    return X_train, X_test, y_train, y_test



def randomforest(X_train, X_test, y_train, y_test):
    #max_depth and min_samples_leaf do not need to be set because our data set is small

    #create classifier object
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    #train classifier object
    clf.fit(X_train, y_train)
    #test classifier
    y_pred = clf.predict(X_test)

    correct_results = np.count_nonzero(y_pred == y_test)

    print(len(y_test))
    print(correct_results)