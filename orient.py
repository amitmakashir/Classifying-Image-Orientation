#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:32:45 2018

@author: rohit amit akshay
"""
'''
We have implemented the three algorithms in three separate files namely:
    1:adaboost.py
    2:knn.py
    3.forest.py
For file reading,writing and transforming operations we have used machinelearning,py file.
'''
from adaboost import *
from knn import *
from forest import *
from adaboost import *
from machinelearning import *
import sys
import os
import numpy as np
#pickle to store the model_file.txt
import pickle
import pandas as pd

# arguments for program
train_flag=sys.argv[1]
ip_file=sys.argv[2]
model_file=sys.argv[3]
model_use=sys.argv[4]

# get current working directory
os.chdir(os.getcwd())

# Load original and transformed feature files based on model request
if train_flag=='train' and model_use!='adaboost':
    y,X = file_load("train-data.txt")
elif train_flag=='train' and model_use=='adaboost':
    y,X = file_load("train-data.txt",False)
elif train_flag=='test' and model_use=='adaboost':
    y_test,X_test = file_load("train-data.txt",False)
elif train_flag=='test' and (model_use=='nearest' or model_use=='best'):
    y,X = file_load("train-data.txt")
    y_test,X_test = file_load("test-data.txt")
else:
    y_test,X_test = file_load("test-data.txt")

#check train flag
if train_flag=='train':
    if model_use=='nearest' or model_use=='best':
        model_file=open('nearest_file.txt','wb')
        pickle.dump(X,model_file)
        model_file.close()
        
    if model_use=='forest':
        model_file = open('forest_file.txt', 'wb')

        sample_size = 2
        trees = 1
        max_depth = 2
        features = "sqrt"

        randfor = RandomForest(trees,sample_size,max_depth,features)
        randfor.fit(X,y)

        pickle.dump(randfor, model_file)
        model_file.close()

    if model_use=='adaboost':
        #generate feature(pixel) combinations based on c_val
        c_val=100
        pixel_combinations=generate_combinations(1,c_val)
        train_sample_1,error_val_1,classified_1,classifier_dict_1=adaboost_calculation(X,y,pixel_combinations,0,90)
        pixel_combinations=generate_combinations(2,c_val)
        train_sample_2,error_val_2,classified_2,classifier_dict_2=adaboost_calculation(X,y,pixel_combinations,0,180)
        pixel_combinations=generate_combinations(3,c_val)
        train_sample_3,error_val_3,classified_3,classifier_dict_3=adaboost_calculation(X,y,pixel_combinations,0,270)
        pixel_combinations=generate_combinations(4,c_val)
        train_sample_4,error_val_4,classified_4,classifier_dict_4=adaboost_calculation(X,y,pixel_combinations,90,180)
        pixel_combinations=generate_combinations(5,c_val)
        train_sample_5,error_val_5,classified_5,classifier_dict_5=adaboost_calculation(X,y,pixel_combinations,90,270)
        pixel_combinations=generate_combinations(6,c_val)
        train_sample_6,error_val_6,classified_6,classifier_dict_6=adaboost_calculation(X,y,pixel_combinations,180,270)    
        model_file=open('adaboost_file.txt','wb')
        pickle.dump(classifier_dict_1,model_file)
        pickle.dump(classifier_dict_2,model_file)
        pickle.dump(classifier_dict_3,model_file)
        pickle.dump(classifier_dict_4,model_file)
        pickle.dump(classifier_dict_5,model_file)
        pickle.dump(classifier_dict_6,model_file)
        model_file.close()

# Else is Testing phase
else:
    if model_use=='adaboost':
        model_file=open('adaboost_file.txt','rb')
        #load model files(dictionaries) from pickle
        model_file.seek(0)
        classifier_dict_1 = pickle.load(model_file)
        classifier_dict_2 = pickle.load(model_file)
        classifier_dict_3 = pickle.load(model_file)
        classifier_dict_4 = pickle.load(model_file)
        classifier_dict_5 = pickle.load(model_file)
        classifier_dict_6 = pickle.load(model_file)
        model_file.close()

        test_sample=pd.DataFrame(X_test)
        y_test=pd.DataFrame(y_test)
        test_sample['weights']=1/test_sample.shape[0]
        test_sample['output']=y_test[0]
        test_sample=np.array(test_sample)

        orientation=[]
        model_list=[[0,90],[0,180],[0,270],[90,180],[90,270],[180,270]]

        dict_model=[classifier_dict_1,classifier_dict_2,classifier_dict_3,classifier_dict_4,classifier_dict_5,classifier_dict_6]
        for index in range(test_sample.shape[0]):
            result_orient={}
            for model_index in range(len(model_list)):
                result=calculate_hypothesis(test_sample[index,:],dict_model[model_index],model_list[model_index][0],model_list[model_index][1])
                if result not in result_orient.keys():
                    result_orient[result]=1
                result_orient[result]+=1
            orientation.append(max(result_orient.items(), key=operator.itemgetter(1))[0])

        count=0
        for i in range(len(orientation)):
            if int(test_sample[i,-1])==orientation[i]:
                count+=1

        generate_output(ip_file,orientation,'output_adaboost.txt')
        print('Accuracy for Adaboost is :',(count/len(orientation))*100)

    elif model_use == "forest":
        model_file = open("forest_file.txt", 'rb')
        randfor = pickle.load(model_file)
        model_file.close()

        predicted_y = randfor.predict(X_test)
        generate_output(ip_file,predicted_y,'output_forest.txt')
        score = randfor.score(predicted_y, y_test)
        print('Accuracy for Random Forest is :',score*100)
        
    elif model_use=='nearest' or model_use=='best':
        c=0
        op=[]
        kvalue=47
        for image_number in range(len(y_test)):
            knearest=(calc_manhattan(y,X,X_test[image_number]))
            current_op=knn_classifier(y,X_test,knearest,kvalue)
            op.append(current_op)
            if y_test[image_number]==current_op:
                c=c+1
                
        f_name='output_'+model_use+'.txt'
        generate_output(ip_file,op,f_name)
        print('Accuracy for ',model_use,' model is :', (c/len(y_test))*100)
        