#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 01:51:19 2018

@author: rohit
"""
#discussed implementation with Raj Thakkar and Srinithish Kandagadla
import numpy as np
import os
import math
import pandas as pd
import operator
import pickle
import time

os.chdir(os.getcwd())

def generate_combinations(k,c_val):
    k = c_val
    np.random.seed(k)
    pixel_combinations=[]
    for k_val in range(k):
        pixel_combinations.append(np.random.randint(0,191,size=2).tolist())
    return pixel_combinations

def adaboost_calculation(train_data,result_train_data,pixel_combinations,limit1,limit2):
    
    train_sample=pd.DataFrame(train_data)
    train_sample['weights']=1/train_sample.shape[0]
    train_sample['output']=result_train_data
    train_sample=train_sample.loc[train_sample['output'].isin([limit1,limit2])]
    train_sample.loc[:, 'output'].replace([limit1,limit2], [-1,1], inplace=True)
    train_sample=np.array(train_sample)

    classifier_dict={}
    
    for combi in pixel_combinations:
        error_val=0
        classified=[]
        predicted=[]
        for train_ind in range(train_sample.shape[0]):
            if train_sample[train_ind,combi[0]]>=train_sample[train_ind,combi[1]]:
                predicted.append(-1)
            else:
                predicted.append(1)
        for train_ind in range(train_sample.shape[0]):
            if train_sample[train_ind,-1]!=predicted[train_ind]:
                error_val+=train_sample[train_ind,-2]
            else:
                classified.append(train_ind)
        if error_val<0.5:
            classifier_dict[(combi[0],'>',combi[1])]=0
                
            f_val=error_val/(1-error_val)
            for element in classified:
                train_sample[element,-2]*=f_val
            
            total_weight=sum(train_sample[:,-2])
            for index in range(train_sample.shape[0]):
                train_sample[index,-2]=train_sample[index,-2]/total_weight
            #print(error_val)
            a_val=math.log((1-error_val)/error_val)
            classifier_dict[(combi[0],'>',combi[1])]=a_val
    
    return train_sample,error_val,classified,classifier_dict

def calculate_hypothesis(train_sample,classifier_dict,limit1,limit2):
    total=0
    for keys,a_val in classifier_dict.items():
        if train_sample[keys[0]]>train_sample[keys[2]]:
            total+=(-1*a_val)
        else:
            total+=(1*a_val)
    if(np.sign(total)<0):
        return limit1
    else:
        return limit2
