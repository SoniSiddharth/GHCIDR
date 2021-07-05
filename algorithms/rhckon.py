from operator import itemgetter
import numpy as np
from .baselineRHC import homogeneousClustering
from utils import *

def RHCKON(datasets, args):
	"""
		KFarthest+RHC algorithm, returns the condensed set
		max_size is a paramter which is used by the progress bar
		K is given as input by the init method
	"""
	print("RHCKON is running:")
	X_train,Y_train = datasets
	if checkClusters(args)==False:
		print("First generating the homogenous clusters and storing it.")
		homogeneousClustering(X_train,Y_train,args)
	Clusters = loadFromPickle("./Clusters/"+args.datasetName+'.pickle')
	CondensedSet = []
	for i in Clusters:
		cImages = i[0]
		meanVector = np.mean(cImages,axis=0)
		distances = []
		for j in cImages:
			distances.append([getL2NormDistnce(meanVector,j),j])# KFarthest
		distances.sort(reverse=True,key=itemgetter(0))
		for j in range(min(args.KFarthest,len(distances))):
			CondensedSet.append([distances[j][1],i[1]]) # put k farthest
		CondensedSet.append([distances[-1][1],i[1]]) # put mean vector of C into CS 
	return CondensedSet