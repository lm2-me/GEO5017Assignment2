

import numpy as np
import matplotlib.pyplot as plt
import os

import Features as ft


### import csv file to read features, remove before sumbitting

#pointcloudsdummy = np.random.randn(800).reshape((100,8))
#pointcloudsdummy = np.random.randint(0,500,3000).reshape((500,6))

cwd = os.getcwd()

print(cwd)

# np.savetxt('features_norm.csv', normalized_object_features, delimiter=',')
dataload = np.loadtxt('features_norm.csv', delimiter=',')

features_only = dataload[:,1:]

print(features_only)

#Main
if __name__ == "__main__":
    ft.pointCloudDirectory = ft.importFiles()
    ft.object_features = np.array(ft.allObjectProperties(pointCloudDirectory))
    ft.normalized_object_features = ft.normalize_features(object_features)




