import numpy as np
from utils import *
import os
def homogeneousClustering(X_train,Y_train,args):
    """
        Centroid based RHC.
        Returns Condensed Set
    """
    print("Making Clusters")
    clusterStore = []
    uniqueClasses = len(np.unique(Y_train))
    imageList,imagesAll = hashedImages(X_train,Y_train)
    while len(imagesAll)>0:
        cImages = imagesAll.pop(0)      # pop front
        checkForHomogenousLabels = labelsHomogenous(cImages,imageList)
        if len(set(checkForHomogenousLabels))==1:
            # If Homogeneous then store cluster
            clusterStore.append([cImages,checkForHomogenousLabels[0]])
        else:
            # If Non Homogeneous then recurse
            classCentroids = getInitCentroids(uniqueClasses,cImages,imageList)
            clusters = KMeans(n_clusters=len(classCentroids), init=classCentroids, n_init=1)
            clusters.fit(np.array(cImages))
            setOfClusters = getClusterDataPoints(clusters.cluster_centers_,cImages)
            for i in setOfClusters.keys():
                imagesAll.append(setOfClusters[i])
    if "Clusters" not in os.listdir():
        os.mkdir("Clusters")
    path = "./Clusters/"+args.datasetName+'.pickle'
    saveAsPickle(clusterStore,path)
    print("Clusters are saved")

def RHC(datasets,args):
    print("RHC started")
    X_train,Y_train = datasets
    if checkClusters(args)==False:
       homogeneousClustering(X_train,Y_train,args)
    Clusters = loadFromPickle("Clusters/"+args.datasetName+".pickle")
    CondensedSet = []
    for cluster in Clusters:
        CondensedSet.append([np.mean(cluster[0],axis=0),cluster[1]])
    print("RHC Done")
    print(len(CondensedSet))
    return CondensedSet
    