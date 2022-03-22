

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import svm

import Features as ft
import Evaluation as ev
import SVM
import RF
# import preprocessing from sklearn
from sklearn import preprocessing



# ### import csv file to read features, remove before sumbitting

# #pointcloudsdummy = np.random.randn(800).reshape((100,8))
# #pointcloudsdummy = np.random.randint(0,500,3000).reshape((500,6))

# cwd = os.getcwd()

# print(cwd)

# # np.savetxt('features_norm.csv', normalized_object_features, delimiter=',')
# dataload = np.loadtxt('features_norm.csv', delimiter=',')

# features_only = dataload[:,1:]
# #print(len(features_only))
# #print(features_only)



#Main
if __name__ == "__main__":
    y_true = np.loadtxt('y_true.csv', dtype='str', delimiter='\n')

    pointCloudDirectory = ft.importFiles()
    object_features = np.array(ft.allObjectProperties(pointCloudDirectory))
    features_only = ft.normalize_features(object_features)
    print(features_only)

    ##dataload = np.loadtxt('features_norm.csv', delimiter=',')

    ## = dataload[:,1:]
    #labeling the data
    labelen = preprocessing.LabelEncoder()
    labelen = labelen.fit_transform(y_true)
    # onehot = preprocessing.OneHotEncoder()
    # label_onehot = onehot.fit_transform(labelen.reshape(-1,1))


    #####
    X_train, X_test, y_train, y_test = RF.splitdata(features_only, y_true)
    y_predRF, y_testRF = RF.randomforest(X_train, X_test, y_train, y_test)
    RF.rf_Plot_max_depth(X_train, X_test, y_train, y_test)
    RF.rf_Plot_n_estiamators(X_train, X_test, y_train, y_test)


    rf_oa = ev.overallAccuracy(y_testRF, y_predRF)
    rf_mpa = ev.meanPerClassAccuracy(y_testRF, y_predRF)
    rf_cm = ev.confusionMatrix(y_testRF, y_predRF)

    print('Random Forest: Overall Accuracy', rf_oa)
    print('Random Forest: Mean Per-Class Accuracy', rf_mpa)
    print('Random Forest: Confusion Matrix \n', rf_cm)


