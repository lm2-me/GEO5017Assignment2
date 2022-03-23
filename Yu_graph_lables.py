import numpy as np
import matplotlib.pyplot as plt

dataload = np.loadtxt('features_norm_new.csv', delimiter=',')

labels = np.arange(0,500)

for i in range(0,500):
    # print(i//100)
    labels[i] = i//100




clusters = labels

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlim(0.4,1)
ax.set_ylim(0,0.8)
ax.set_zlim(0, 0.8)

for m, c, a in [('o',0, 0.5), ('^',1, 1), ('*',2, 1),('s',3, 1),('p',4, 1)]:
    xs = [dataload[i,6] for i, num_value in enumerate(dataload[:,0]) if int(clusters[i]) == int(c)]
    ys = [dataload[i,2] for i, num_value in enumerate(dataload[:,0]) if int(clusters[i]) == int(c)]
    zs = [dataload[i,0] for i, num_value in enumerate(dataload[:,0]) if int(clusters[i]) == int(c)]
    ax.scatter(xs, ys, zs, marker=m, alpha=a)

ax.set_xlabel('Squareness')
ax.set_ylabel('Average Height')
ax.set_zlabel('Height')

plt.show()

# height, volume, avg_height, area, ratio, num_planes, squareness - these are the feature numbers, which can be edited above in (dataload[i,x]) in the xs, ys, and zs.
# Building, car, fence, pole, tree 
# blue transparent points: buildings. Orange triangles: cars. Green stars: Fence. Red squares: poles. Purple points: trees






clusters = labels
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_zlim(0, 0.3)
ax.set_ylim(0,0.3)
ax.set_xlim(0,0.3)

for m, c, a in [('o',0, 0.5), ('^',1, 1), ('*',2, 1),('s',3, 1),('p',4, 1),('P',5,1)]:
    xs = [dataload[i,1] for i, num_value in enumerate(dataload[:,0]) if int(clusters[i]) == int(c)]
    ys = [dataload[i,3] for i, num_value in enumerate(dataload[:,0]) if int(clusters[i]) == int(c)]
    zs = [dataload[i,4] for i, num_value in enumerate(dataload[:,0]) if int(clusters[i]) == int(c)]
    ax.scatter(xs, ys, zs, marker=m, alpha=a)

ax.set_xlabel('Volume')
ax.set_ylabel('Area')
ax.set_zlabel('Ratio')

plt.show()

# height, volume, avg_height, area, ratio, num_planes, squareness - these are the feature numbers, which can be edited above in (dataload[i,x]) in the xs, ys, and zs.
# Building, car, fence, pole, tree 
# blue transparent points: buildings. Orange triangles: cars. Green stars: Fence. Red squares: poles. Purple points: trees