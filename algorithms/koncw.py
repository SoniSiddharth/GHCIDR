from operator import itemgetter
import numpy as np
from .baselineRHC import homogeneousClustering
from utils import *

def KONCW(datasets,args):
    print("KONCW is running:")
    print("First generating the homogenous clusters and storing it.")
    X_train,Y_train = datasets
    if checkClusters(args)==False:
        homogeneousClustering(X_train,Y_train,args)
    Clusters = loadFromPickle("./Clusters/"+args.datasetName+".pickle")
    print("Applying weights and selecting important images.")
    CondensedSet = []
    for i in Clusters:
        cImages = i[0]
        meanVector = np.zeros(cImages[0].shape)
        for j in cImages:
            meanVector+=j
        meanVector = meanVector/len(cImages)
        distances = []
        dis = []
        for j in cImages:
            r = getL2NormDistnce(meanVector,j)
            distances.append([r,j])
            dis.append(r)
        distances.sort(key=itemgetter(0),reverse=True)
        select = int(max((1-args.alpha)*len(cImages),1))
        for k in distances[:select]:
            CondensedSet.append([k[1],i[1]])
        CondensedSet.append([distances[-1][1],i[1]])
    return CondensedSet