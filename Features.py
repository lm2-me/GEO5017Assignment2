#Github
''' git status, git add ., git commit -m "comment about update", git push. --> git pull'''

#Import Point Cloud
import numpy as np
import open3d as o3d
import math as m

from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt


print("start program for importing files")

import os

def importFiles():
    cwd = os.getcwd() # get current directory of file
    #print("cwd = " + cwd)
    filewd = (cwd) # tuple to lock original folder location when changing directory
    # print("this is cwd: ", cwd) # check
    # print("this is filewd: ", filewd) # check

    pointcloudFolder = filewd + "\pointclouds"
    #print(pointcloudFolder) #To check

    os.chdir(pointcloudFolder)
    print("new directory changed to : {0}".format(os.getcwd()))

    d = {}
    """the list of point clouds are being put in a dictionary as numpy arrays - not sure if this is the best.
    I also considered Pandas. Also wasn't sure whether a list of list would be annoying.
    This way we can call them specificaly by name, so that might be the easiest when iterating through them.
    """
    pc_names = []
    for i in range(500):
        # print(i)
        number = "00" + str(i)
        three_digit = number[-3:]
        open_array = np.genfromtxt("{0}.xyz".format(three_digit))
        d["pc{0}".format(three_digit)] = open_array
        pc_names.append("pc{0}".format(three_digit))
    
    cwd = os.getcwd()
    filewd = (cwd[: len(cwd) - 11])
    #print('file wd' + filewd)
    save_pc = filewd + 'pointclouds.txt'

    with open (save_pc, 'w') as f:
        for i in range(500):
            f.write(str(d[pc_names[i]]) + "\n")
    f.close()

    #print(d["pc385"])
    return d

#visualize point cloud
def visualizePC(pc):
    cloud = currento3dPCfile(pc)
    o3d.visualization.draw_geometries([cloud])

#current o3d point cloud gets the file for open3d to use in visualizations
def currento3dPCfile(pc):

    number = "00" + str(pc)
    three_digit = number[-3:] 
    filewd = os.getcwd()
    folder = "{0}\{1}.xyz".format(filewd, three_digit)

    currentPointCloud = o3d.io.read_point_cloud(folder)
    return currentPointCloud

#Get object features for each point cloud height
def allObjectProperties(pointCloudDirectory):
    i = 0
    object_features = []
    pc_list = list(pointCloudDirectory)[:20]
    print('Evaluating point cloud features')
    for pc in pc_list:
        print ('now working on point cloud', str(pc), end="\r")

        ###remove before submitting

        #Get current point cloud
        currentPointCloud = currentPC(pointCloudDirectory, pc)
        currentPointCloud_o3d = currento3dPCfile(pc)

        #Visualize Point Cloud
        #visualizePC(pc)

        
        #Get properties by calling related function, only getting 3 best features after analysis of all features
        height = objectHeight(currentPointCloud)
        volume = convexHull(currentPointCloud_o3d)
        avg_height = objectAverageHeight(currentPointCloud)
        area, ratio = areaBase(currentPointCloud_o3d)
        planarity = planarityPC(currentPointCloud_o3d)
        rectangle_deviation = rectangleDeviation(currentPointCloud_o3d)

        object_features.append([height, volume, avg_height, area, ratio, rectangle_deviation])
        #object_features.append([height, avg_height, num_planes])
        
        i += 1
        #print(str(i) + " height: " + str(height) + " volume: " + str(volume) + " average height: " + str(avg_height) + " area: " + str(area) + " ratio: " + str(ratio) + " num planes: " + str(num_planes))

    return object_features

#Get current point cloud and save to new array
def currentPC(pointCloudDirectory, pc):
    number = "00" + str(pc)
    three_digit = number[-3:]
    name = "pc" + three_digit
    currentPointCloud = pointCloudDirectory[name]
    return currentPointCloud

#normalize  to put into range from 0 to 1
def normalize_features(object_features):
    print('Normalizing point cloud features')
    print(object_features)
    all_normalized_features = np.copy(object_features)
    for i in range(0, object_features.shape[1]):
        min = np.min(object_features[:,i])
        normalized_feature = object_features[:,i] - min
        max = np.max(normalized_feature)
        normalized_feature = normalized_feature / max
        all_normalized_features[:,i] = normalized_feature
    return all_normalized_features

def rectangleDeviation(allpoints):

    allpoints_arrayZero = np.asarray(allpoints.points)
    for point in allpoints_arrayZero:
        point[2] = 0
    
    allpoints_arrayOne = allpoints_arrayZero.copy()

    for point in allpoints_arrayOne:
        point[2] = 1

    allpoints_array = np.concatenate((allpoints_arrayZero,allpoints_arrayOne))
    
    flat_pc = o3d.geometry.PointCloud()
    flat_pc.points = o3d.utility.Vector3dVector(allpoints_array)

    
    convhull, _ = flat_pc.compute_convex_hull()
    convhull_lns = o3d.geometry.LineSet.create_from_triangle_mesh(convhull)
    convhull_lns.paint_uniform_color((0, 0, 1))

    twoD_volume = convhull.get_volume()

    bBox = flat_pc.get_axis_aligned_bounding_box()
    bBox.color = (0, 0, 1)

    min = bBox.get_min_bound()
    max = bBox.get_max_bound()

    length = max[0] - min[0]
    width = max[1] - min[1]

    bBox_area = length * width   
    
    rectangle_deviation = twoD_volume / bBox_area

    #visualize convex hull
    #o3d.visualization.draw_geometries([flat_pc, convhull,bBox])

    return rectangle_deviation

#Get feature 1: Height
def objectHeight(currentPointCloud):
    maxZ = 0
    minZ = np.inf

    for point in currentPointCloud:
        if point[2] > maxZ:
            maxZ = point[2]
        #maxZ now largest Z
    
    for point in currentPointCloud:
        if point[2] < minZ:
            minZ = point[2]
        #minZ now smallest Z

    height = maxZ - minZ
    return height

def objectMaxZ(currentPointCloud):
    maxZ = 0

    for point in currentPointCloud:
        if point[2] > maxZ:
            maxZ = point[2]
        #maxZ now largest Z

    height = maxZ
    return height

#Feature 2: Convex hull for Volume
def convexHull(pc):
    convhull, _ = pc.compute_convex_hull()
    convhull_lns = o3d.geometry.LineSet.create_from_triangle_mesh(convhull)
    convhull_lns.paint_uniform_color((0, 0, 1))

    # euler = convhull.euler_poincare_characteristic()
    #visualize convex hull
    #o3d.visualization.draw_geometries([pc, convhull_lns])
    
    volume = convhull.get_volume()
    return volume

#Feature 3: planarity
def planarityPC(pc):
    allpoints = pc
    numplanes = 0
    #get planes with o3d segment_plane
    
    pcarray = np.asarray(pc.points)
    numpoints = len(pcarray)

    #accuracy
    p = 0.99
    #error rate
    e = 0.9
    #percision level
    s = 3
    #iterations
    n = int(m.ceil(m.log(1 - p) / m.log(1 - m.pow(1 - e, s))))

    while len(np.asarray(allpoints.points)) > .2 * len(pcarray):
        indexes = []
        
        plane_model, inliers = allpoints.segment_plane(distance_threshold=0.1,ransac_n=3, num_iterations=n)

        inlier_cloud = allpoints.select_by_index(inliers)


        for item in range(len(np.asarray(allpoints.points))):
            if item in inliers:
                continue
            else:
                indexes.append(item)

        allpoints = allpoints.select_by_index(indexes)
        numplanes += 1

        #visualize planes
        #inlier_cloud.paint_uniform_color([1.0, 0, 0])
        #allpoints.paint_uniform_color([0, 0, 1])
        #o3d.visualization.draw_geometries([allpoints, inlier_cloud])

    return numplanes

#Get feature 4: Average Height
def objectAverageHeight(currentPointCloud):
    
    npCurrentPointCloud = np.array(currentPointCloud)
    allHeights = npCurrentPointCloud[:,2]

    averageHeight = sum(allHeights) / len(allHeights)

    return averageHeight

#Get feature 5: Area of plan view
def areaBase(pc):
    allpoints_arrayZero = np.asarray(pc.points)
    for point in allpoints_arrayZero:
        point[2] = 0
    
    allpoints_arrayOne = allpoints_arrayZero.copy()

    for point in allpoints_arrayOne:
        point[2] = 1

    allpoints_array = np.concatenate((allpoints_arrayZero,allpoints_arrayOne))
    
    flat_pc = o3d.geometry.PointCloud()
    flat_pc.points = o3d.utility.Vector3dVector(allpoints_array)

    
    convhull, _ = flat_pc.compute_convex_hull()
    convhull_lns = o3d.geometry.LineSet.create_from_triangle_mesh(convhull)
    convhull_lns.paint_uniform_color((0, 0, 1))

    area = convhull.get_volume()

    bBox = pc.get_axis_aligned_bounding_box()
    bBox.color = (0, 0, 1)

    min = bBox.get_min_bound()
    max = bBox.get_max_bound()

    length = max[0] - min[0]
    width = max[1] - min[1]

    ratio = length / width
    
    #visualize bounding box
    #o3d.visualization.draw_geometries([pc, bBox])
    return area, ratio

#Feature 7: proportion of plan to bounding box
def rectangleDeviation(allpoints):

    allpoints_2D = np.asarray(allpoints.points)
    allpoints_2D = np.delete(allpoints_2D, 2, 1)

    hull = ConvexHull(allpoints_2D)

    twoD_volume = hull.volume

    allpoints_arrayZero = np.asarray(allpoints.points)
    for point in allpoints_arrayZero:
        point[2] = 0

    #Visualize Flat Point Cloud
    # flatZero_pc = o3d.geometry.PointCloud()
    # flatZero_pc.points = o3d.utility.Vector3dVector(allpoints_arrayZero)
    # o3d.visualization.draw_geometries([flatZero_pc])

    allpoints_arrayOne = allpoints_arrayZero.copy()

    for point in allpoints_arrayOne:
        point[2] = 1

    allpoints_array = np.concatenate((allpoints_arrayZero,allpoints_arrayOne))

    flat_pc = o3d.geometry.PointCloud()
    flat_pc.points = o3d.utility.Vector3dVector(allpoints_array)

    # #for visualizing volume vs boundign box
    # convhull_flat, _ = flat_pc.compute_convex_hull()
    # convhull_flat_lns = o3d.geometry.LineSet.create_from_triangle_mesh(convhull_flat)
    # convhull_flat_lns.paint_uniform_color((0, 1, 1))

    bBox_flat = flat_pc.get_axis_aligned_bounding_box()
    bBox_flat.color = (1, 0, 0)

    bBox_volume = bBox_flat.volume()
    rectangle_deviation = twoD_volume / bBox_volume

    #visualize convex hull and bounding box
    # convhullbbox = convhull_flat.get_axis_aligned_bounding_box().volume()
    #o3d.visualization.draw_geometries([flat_pc, convhull_flat_lns, bBox_flat])

    return rectangle_deviation