

import numpy as np
import matplotlib.pyplot as plt
import os

import Features as ft
import Evaluation as ev
import RF


### import csv file to read features, remove before sumbitting

#pointcloudsdummy = np.random.randn(800).reshape((100,8))
#pointcloudsdummy = np.random.randint(0,500,3000).reshape((500,6))

cwd = os.getcwd()

print(cwd)

# np.savetxt('features_norm.csv', normalized_object_features, delimiter=',')
dataload = np.loadtxt('features_norm.csv', delimiter=',')

features_only = dataload[:,1:]
#print(len(features_only))
#print(features_only)

y_true = np.loadtxt('y_true.csv', dtype='str', delimiter='\n')

#Main
if __name__ == "__main__":
    #ft.pointCloudDirectory = ft.importFiles()
    #ft.object_features = np.array(ft.allObjectProperties(ft.pointCloudDirectory))
    #ft.normalized_object_features = ft.normalize_features(ft.object_features)

    X_train, X_test, y_train, y_test = RF.splitdata(features_only)
    y_predRF, y_testRF = RF.randomforest(X_train, X_test, y_train, y_test)

    rf_oa = ev.overallAccuracy(y_testRF, y_predRF)
    rf_mpa = ev.meanPerClassAccuracy(y_testRF, y_predRF)
    rf_cm = ev.confusionMatrix(y_testRF, y_predRF)

    print('Random Forest: Overall Accuracy', rf_oa)
    print('Random Forest: Mean Per-Class Accuracy', rf_mpa)
    print('Random Forest: Confusion Matrix \n', rf_cm)


