from operator import itemgetter
import numpy as np
from .baselineRHC import homogeneousClustering
from utils import *

def CWKC(datasets,args):
    print("CWKC is running:")
    X_train,Y_train = datasets
    if checkClusters(args)==False:
        print("First generating the homogenous clusters and storing it.")
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
        if select==1:
            CondensedSet.append([distances[0][1],i[1]])
        else:
            temp = np.zeros(cImages[0].shape)
            myset = [distances[0][1],distances[-1][1]]
            for k in range(select-2):
                mini = np.inf
                point = myset[0]
                maxi = 0
                for l in cImages:
                    for p in myset:
                        if (l-p==temp).all()==False:
                            mini = min(getL2NormDistnce(l,p),mini)
                    if maxi<mini:
                        maxi = mini
                        point = l
                myset.append(l)
            for k in myset:
                CondensedSet.append([k,i[1]])
    return CondensedSet