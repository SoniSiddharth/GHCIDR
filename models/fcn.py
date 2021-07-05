from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from utils import *
import argparse
import os,sys

def define_model():
	global args
	model = Sequential()
	model.add(Dense(512, input_shape=args.imageSize))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(args.noOfClasses))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model 

def run_test_harness():
	global args
	trainX, trainY, testX, testY = args.getOriginalData(mode=1)
	if(args.fullDataset=='No'):
		trainX = []
		trainY = []
		for i in args.reducedData:
			trainX.append(i[0])
			trainY.append(i[1])
		trainX = np.array(trainX, dtype="float32")
		trainY = np.array(trainY, dtype="float32")
	trainY = np_utils.to_categorical(trainY, args.noOfClasses)
	model = define_model()    
	print(trainX.shape)
	print(testX.shape)
	print(trainY.shape)
	print(testY.shape)
	history = model.fit(trainX, trainY, batch_size=args.batchSize, epochs=args.epochs,verbose=1)
	# evaluate model
	_, acc = model.evaluate(testX, testY)
	print('> %.3f' % (acc * 100.0))

parser = argparse.ArgumentParser()
parser.add_argument("-epochs", default='100', type=str)
parser.add_argument("-datasetName", default='MNIST', type=str)
parser.add_argument("-variantName", default='RHC', type=str)
parser.add_argument("-lr", default='0.001', type=float)
parser.add_argument("-batchSize", default=64, type=int) 
parser.add_argument("-fullDataset", default='No', type=str) 
args = parser.parse_args()
Datasets = {
			"MNIST":[getMnistData,(784,1),10],
			"FMNIST":[getFMnistData,(784,1),10] 
			}
# args.uniqueClasses
getOriginalData, imageSize, noOfClasses = Datasets[args.datasetName]
path = "../datasetPickle/" +args.datasetName + '_' + args.variantName +".pickle"
reducedData = loadFromPickle(path)
args.getOriginalData = getOriginalData
args.imageSize = imageSize
args.noOfClasses = noOfClasses
args.reducedData = reducedData
run_test_harness()