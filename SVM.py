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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    return X_train, X_test, y_train, y_test



def support_vector_machine(X_train, X_test, y_train, y_test):

    #create classifier object
    classifier = svm.SVC(kernel='linear', C=1000)
    #train classifier object
    classifier.fit(X_train,y_train)
    #test classifier
    y_pred = classifier.predict(X_test)

    return y_pred, y_test