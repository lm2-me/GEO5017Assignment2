import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

D = np.loadtxt('features_norm_new.csv', delimiter=',')

transformer = RobustScaler().fit(D)

D = transformer.transform(D)



fig, ax = plt.subplots()

vp = ax.violinplot(D, widths=1,
                   showmeans=False, showmedians=False, showextrema=False)
# styling:
for body in vp['bodies']:
    body.set_alpha(0.9)
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(-2, 3), yticks=np.arange(-2, 4))

# height = objectHeight(currentPointCloud)
#         volume = convexHull(currentPointCloud_o3d)
#         avg_height = objectAverageHeight(currentPointCloud)
#         area, ratio = areaBase(currentPointCloud_o3d)
#         num_planes = planarityPC(currentPointCloud_o3d)

plt.xticks(ticks=[1,2,3,4,5,6,7],labels=['height','volume','avg_height','area','ratio','planarity', 'squareness'], rotation=-30)
plt.show()




y_true = np.loadtxt('y_true.csv', dtype=str)

X = D
y = y_true


test_size_neg = []
test_size_record = []
AO = []

test_size = 0
while test_size < 0.99:
    test_size += 0.01
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    classifier = svm.SVC(kernel='rbf', C=1000,decision_function_shape='ovr', max_iter= -1)
    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    test_size_record.append(test_size)
    AO.append(accuracy)
    test_size_neg.append(1-test_size)
    # print(test_size)


import matplotlib.pyplot as plt

plt.scatter(test_size_neg, AO)
plt.xlabel("Train data size")
plt.ylabel("Overall Accuracy")
plt.show()
