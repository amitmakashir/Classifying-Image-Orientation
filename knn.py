#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:59:41 2018

@author: Rathi
"""

import numpy as np
import operator
import time
import os

os.chdir(os.getcwd())

#Knn reads input files from machinelearning.py
from machinelearning import file_load

#labels_train = Labels of 36976 training images.
#train_data = Pixel values of all these 36976 training images.
#labels_test = Labels of all test images (943 images in given test file). This is used to check accuracy.
#test_data = Pixel values of all test images.

# labels_train,train_data=file_load('train-data.txt')
# labels_test,test_data=file_load('test-data.txt')

# Calculating Euclidean distance of each test image with respect to all training images.
# sqrt((x-x1)^2 + (x-x2)^2 + (x-x3)^2 + .....)
# We ignore taking square root here as this value nullifies at the end.
# This function returns a list with euclidean distance of 1 test image wrt every training image and their respective training labels.
def calc_euclidean(labels_train,train_data,test_image):   
    knn=[]
    for i in range(train_data.shape[0]):        
        dist=np.sum(np.square(np.subtract(train_data[i],test_image)))
        knn.append([dist,labels_train[i]])    
    return knn

# Calculating Manhattan distance of each test image with respect to all training images.
# |x-x1| + |x-x2| + ....
# Where x are pixel values of test image and x1,x2,x3,...are pixel values of 1st,2nd,3rd...training images.
def calc_manhattan(labels_train,train_data,test_image):
    knn=[]
    for i in range(train_data.shape[0]):
        dist=np.sum(np.abs(np.subtract(train_data[i],test_image)))
        knn.append([dist,labels_train[i]])
    return knn

# Classifies the test image with respect to k-nearest neighbours.
# This is done by initially sorting the distance list obtained and considering least k-distances.
def knn_classifier(lt,tt,knearest,kvalue):
    degrees=[]
    knearest.sort(key=operator.itemgetter(0))
    for i in range(kvalue):      
        degrees.append(knearest[i][1])       
    return (max(degrees,key=degrees.count))
