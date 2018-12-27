#!/usr/bin/env python3
import numpy as np
import os
import sys

def file_load(filename,transform_flag=True):
    with open(os.getcwd()+"/"+filename) as trainFile:
        lines = [line.split() for line in trainFile]
        
    lines = np.array(lines)
    result_data = lines[:,1].astype(int)
    train_data = np.array(lines[:,2:].astype(int))
    new_train_data = []
    for train_index,x in enumerate(train_data):
        new_train_data.append(np.array(transform(x)))

    new_train_data = np.array(new_train_data)

    if transform_flag:
        return result_data,new_train_data
    else:
        return result_data,np.concatenate((train_data,new_train_data),axis=1) 


def generate_output(train_file,y_pred,op_file):
    try:
        with open(os.getcwd() + "/" + train_file) as trainFile:
            image_names = [line.split()[0] for line in trainFile]

        filename = op_file
        f = open(filename, 'w')
        for i in range(len(y_pred)):
            f.write( image_names[i] + " " + str(y_pred[i]) + "\n")
        f.close()
        return True
    except:
        return False


def transform(x):
    # print(x)
    # print("\n")
    '''
    For top to bottom - rows
    '''
    red_pixels = [[x[j + i] for i in range(0, 24, 3)] for j in range(0, 192, 24)]
    green_pixels = [[x[j + i] for i in range(0, 24, 3)] for j in range(1, 192, 24)]
    blue_pixels = [[x[j + i] for i in range(0, 24, 3)] for j in range(2, 192, 24)]
    # red_pixels = np.array(red_pixels)
    # green_pixels = np.array(green_pixels)
    # blue_pixels = np.array(blue_pixels)

    '''
    Mean average of top,right,bottom,left two rows
    '''
    blue_threshold = 200
    red_threshold = 1

    is_top_blue = [0]
    if np.mean(blue_pixels[0] + blue_pixels[1]) > blue_threshold:
        is_top_blue = [1]

    is_bottom_blue = [0]
    if np.mean(blue_pixels[6] + blue_pixels[7]) > blue_threshold:
        is_bottom_blue = [1]

    is_top_red = [0]
    if np.mean(red_pixels[0] + red_pixels[1]) > red_threshold:
        is_top_red = [1]

    is_bottom_red = [0]
    if np.mean(red_pixels[6] + red_pixels[7]) > red_threshold:
        is_bottom_red = [1]


    red_diff_means_v = [(np.mean(red_pixels[i]) - np.mean(red_pixels[7 - i]))*(4-i) for i in range(4)]
    green_diff_means_v = [(np.mean(green_pixels[i]) - np.mean(green_pixels[7 - i]))*(4-i) for i in range(4)]
    blue_diff_means_v = [(np.mean(blue_pixels[i]) - np.mean(blue_pixels[7 - i]))*(4-i) for i in range(4)]

    '''
    For left to right - columns
    '''
    red_pixels = [[x[j + i] for i in range(0, 192, 24)] for j in range(0, 24, 3)]
    green_pixels = [[x[j + i] for i in range(1, 192, 24)] for j in range(0, 24, 3)]
    blue_pixels = [[x[j + i] for i in range(2, 192, 24)] for j in range(0, 24, 3)]
    # red_pixels = np.array(red_pixels)
    # green_pixels = np.array(green_pixels)
    # blue_pixels = np.array(blue_pixels)

    is_left_blue = [0]
    if np.mean(blue_pixels[0] + blue_pixels[1]) > blue_threshold:
        is_left_blue = [1]

    is_right_blue = [0]
    if np.mean(blue_pixels[6] + blue_pixels[7]) > blue_threshold:
        is_right_blue = [1]

    is_left_red = [0]
    if np.mean(red_pixels[0] + red_pixels[1]) > red_threshold:
        is_left_red = [1]

    is_right_red = [0]
    if np.mean(red_pixels[6] + red_pixels[7]) > red_threshold:
        is_right_red = [1]


    red_diff_means_h = [(np.mean(red_pixels[i]) - np.mean(red_pixels[7 - i]))*(4-i) for i in range(4)]
    green_diff_means_h = [(np.mean(green_pixels[i]) - np.mean(green_pixels[7 - i]))*(4-i) for i in range(4)]
    blue_diff_means_h = [(np.mean(blue_pixels[i]) - np.mean(blue_pixels[7 - i]))*(4-i) for i in range(4)]


    features = red_diff_means_v + green_diff_means_v + blue_diff_means_v + red_diff_means_h + green_diff_means_h + blue_diff_means_h
    features += is_top_blue + is_bottom_blue + is_top_red + is_bottom_red + is_left_blue + is_right_blue + is_left_red + is_right_red
    # print(features)
    # sys.exit()
    return features


#
#
# y_pred = [1]*943
# generate_output("train-data.txt",y_pred)