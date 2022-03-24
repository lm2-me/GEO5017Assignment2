

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import svm
import SVM
import Features as ft
import Evaluation as ev
import Yu_SVM as svm
import Yu_graph_lables as svm_graph
import RF
# import preprocessing from sklearn
from sklearn import preprocessing

#save features to file
# cwd = os.getcwd()
# print(cwd)
# # np.savetxt('features_norm.csv', normalized_object_features, delimiter=',')

#Main
if __name__ == "__main__":
    y_true = np.loadtxt('y_true.csv', dtype='str', delimiter='\n')

    # object_features = np.array(ft.allObjectProperties(pointCloudDirectory))
    # features_only = ft.normalize_features(object_features)
    # print(features_only)
    svm
    svm_graph
    svm.graph_svm()

    #labeling the data
    labelen = preprocessing.LabelEncoder()
    labelen = labelen.fit_transform(y_true)
    # onehot = preprocessing.OneHotEncoder()
    # label_onehot = onehot.fit_transform(labelen.reshape(-1,1))
    object_features = np.loadtxt('features_norm_new.csv', delimiter=',')
    pointCloudDirectory = ft.importFiles()
    # object_features = np.array(ft.allObjectProperties(pointCloudDirectory))
    # features_only = ft.normalize_features(object_features)
    # print(features_only)

    ### uncomment before submitting
    # pointCloudDirectory = ft.importFiles()
    # object_features = ft.allObjectProperties(pointCloudDirectory)


    ## Delete unused features at end
    #Average Height, Squareness [2, 6]
    #object_features2 = object_features[:,[0,2,6]]
    #Height, Planarity, Squareness [0, 5, 6]
    object_features3 = object_features[:,[0,5,6]]
    #Average Height, Planarity, Squareness [2, 5, 6]
    #object_features4 = object_features[:,[2,5,6]]
    #Average Height, Ratio, Squareness [2, 4, 6]
    #object_features5 = object_features[:,[2,4,6]]
    #Height, Average Height, Planarity, Squareness [0, 2, 5, 6]
    #object_features6 = object_features[:,[0,2,5,6]]
    #Height, Average Height, Ratio, Planarity, Squareness  [0, 2, 4, 5, 6]
    #object_features7 = object_features[:,[0,2,4,5,6]]
    #Height, Average Height, Area, Ratio, Planarity, Squareness  [0, 2, 3, 4, 5, 6]
    #object_features7 = object_features[:,[0,2,3,4,5,6]]

       
    #Random Forest
    X_train, X_test, y_train, y_test = RF.splitdata(object_features3, y_true)
    y_predRF, y_testRF = RF.randomforest(X_train, X_test, y_train, y_test)
    #RF.rf_Plot_max_depth(X_train, X_test, y_train, y_test)
    #RF.rf_Plot_n_estiamators(X_train, X_test, y_train, y_test)


    rf_oa = ev.overallAccuracy(y_testRF, y_predRF)
    rf_mpa = ev.meanPerClassAccuracy(y_testRF, y_predRF)
    rf_cm = ev.confusionMatrix(y_testRF, y_predRF)

    print('Random Forest: Overall Accuracy', rf_oa)
    print('Random Forest: Mean Per-Class Accuracy', rf_mpa)
    print('Random Forest: Confusion Matrix \n', rf_cm)

    #Visualize Learning Curve
    RF.learningcurve(object_features3, y_true)

    #SVM
    X_trainSVM, X_testSVM, y_trainSVM, y_testSVM = SVM.splitdata(object_features3, y_true)
    y_predSVM, y_testSVM = SVM.support_vector_machine(X_trainSVM, X_testSVM, y_trainSVM, y_testSVM)

    svm_oa = ev.overallAccuracy(y_testSVM, y_predSVM)
    svm_mpa = ev.meanPerClassAccuracy(y_testSVM, y_predSVM)
    svm_cm = ev.confusionMatrix(y_testSVM, y_predSVM)

    print('Support Vector Machine: Overall Accuracy', svm_oa)
    print('Support Vector Machine: Mean Per-Class Accuracy', svm_mpa)
    print('Support Vector Machine: Confusion Matrix \n', svm_cm)

