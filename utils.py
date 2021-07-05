#impot libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.cluster import KMeans
from keras.datasets import mnist,cifar10,fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation
from operator import itemgetter

def loadFromPickle(path):
    try:
        dbfile = open(path, 'rb')      
    except:
        return None, False
    db = pickle.load(dbfile)
    return db

def saveAsPickle(dataset, path):   
    dbfile = open(path, 'wb') 
    pickle.dump(dataset, dbfile)
    print("Saved at: "+ path)                      
    dbfile.close()

def checkClusters(args):
    if "Clusters" not in os.listdir():
        return False
    path = args.datasetName+'.pickle'
    if path not in os.listdir("Clusters"):
        return False
    return True

def getL2NormDistnce(v1,v2,norm=2):
    """
        L norm between v1 and v2 vectors
        ora is the order of L norm.
    """
    distance = np.linalg.norm(v1-v2,ord=norm)
    return distance

def getClusterDataPoints(listOfClusterCenters,dataX):
    """
        After getting the final centres of the clusters, this function is called
        For every point, this assigns it to its nearest centre.
        return list of clusters
    """
    uniqueClusterDataX = {}
    for clusterCenter in listOfClusterCenters:
        uniqueClusterDataX[tuple(clusterCenter)] = []
    for dataPoint in dataX:
        centerIntial = listOfClusterCenters[0]
        minimumDistance = np.inf
        for centerPoint in listOfClusterCenters:
            distance = getL2NormDistnce(centerPoint,dataPoint)
            if distance<minimumDistance:
                minimumDistance = distance
                centerIntial = centerPoint
        uniqueClusterDataX[tuple(centerIntial)].append(dataPoint)
    # print(len(uniqueClusterDataX))
    llist=[]
    for i in uniqueClusterDataX.keys():
        if(len(uniqueClusterDataX[i])==0):
            llist.append(i)
    for i in llist:
        del uniqueClusterDataX[i]
    return uniqueClusterDataX

def getMnistData(mode=0):
    """
        returns preprocessed MNIST dataset
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784).astype('float32') # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
    # X_test = X_test.reshape(10000, 784).astype('float32')   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.
    X_test = np.array(X_test,dtype="float32")
    X_train /= 255                        # normalize each value for each pixel for the entire vector for each input
    X_test /= 255
    uniqueClasses = 10
    Y_train = y_train
    Y_test = np_utils.to_categorical(y_test, uniqueClasses)
    if mode==1:
        return X_train,Y_train,X_test,Y_test
    return X_train, Y_train

def getCifar10Data(mode=0):
    """
        returns preprocessed CIFAR10 dataset
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(50000, 3072).astype('float32') # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
    # X_test = X_test.reshape(10000, 3072).astype('float32')   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.
    X_test = np.array(X_test,dtype="float32")
    X_train /= 255                                           # normalize each value for each pixel for the entire vector for each input
    X_test /= 255
    uniqueClasses = 10
    Y_train = y_train.tolist()
    Y_train = [i[0] for i in Y_train]
    Y_test = np_utils.to_categorical(y_test, uniqueClasses)
    if(mode==1):
        return X_train,Y_train,X_test,Y_test
    return X_train, Y_train

def getFMnistData(mode=0):
    """
        returns preprocessed MNIST dataset
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(60000, 784).astype('float32') # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
    # X_test = X_test.reshape(10000, 784).astype('float32')   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.
    X_test = np.array(X_test,dtype="float32")
    X_train /= 255                        # normalize each value for each pixel for the entire vector for each input
    X_test /= 255
    uniqueClasses = 10
    Y_train = y_train
    Y_test = np_utils.to_categorical(y_test, uniqueClasses)
    if mode==1:
        return X_train,Y_train,X_test,Y_test
    return X_train, Y_train

def hashedImages(X_train,Y_train):
    imageList = {}
    for i in range(len(X_train)):
        imageList[X_train[i].tobytes()] = Y_train[i]  # image label mapping
    imagesAll = []
    imagesAll.append(X_train)     # images enqueue
    return imageList, imagesAll

def labelsHomogenous(cImages,imageList):
    checkForHomogenousLabels = []           # find if homogenous
    for img in cImages:    
        checkForHomogenousLabels.append(imageList[img.tobytes()])
    return checkForHomogenousLabels
        
def getInitCentroids(uniqueClasses,cImages,imageList):
    classCentroids = []   # all centroids
    uniqueCluster = {}           # unique
    for iLabel in range(uniqueClasses):   
        uniqueCluster[iLabel] = []
    for i in cImages:           # separate on basis of labels
        uniqueCluster[imageList[i.tobytes()]].append(i)
    for i in uniqueCluster.keys(): # find centroids of all classes
        if uniqueCluster[i]:
            meanVector = np.zeros(uniqueCluster[i][0].shape)
            for j in uniqueCluster[i]:
                meanVector+=j
            classCentroids.append(meanVector/len(uniqueCluster[i]))
    classCentroids = np.array(classCentroids)
    return classCentroids
