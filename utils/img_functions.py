########### Utils for datasets ###########

import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
from torchvision import transforms

## Loading images from datasets

def load_single_dataset(origin,csv,column,path) :

    """
    Load either all BV or all ET images.

    Parameter :
        origin : 0 for BV, any other for ET
        csv : path of csv file containing all image filenames with corresponding labels
        column : name of filenames column, 'ImageID' for ET and 'New_name' for BV
        path : path of directory to load all images from
    ------------------------
    Output :
        array : all loaded images
        dataframe : image filenames with matching labels
    """

    resize_shape = (224,224)

    img_list = [] # images are temporarily stored in a list
    # open csv as dataframe
    df_labels = pd.read_csv(csv,sep=';')
    # read filename from the dataframe and load corresponding image
    for i in range(df_labels.shape[0]):
        f = df_labels[column][i]
        try:
            img = Image.open(os.path.join(path,f))
            if origin:
                img = remove_pad_single(img) # remove padding of ET images
            img = img.resize(resize_shape) # resize
            img_list.append(img)
        except Exception as e:
            print(e)
    return np.array(img_list), df_labels

def load_all(csvBV, imgsBV, csvET, imgsET):

    """
    Load all BV and ET images.
    
    Note that ET dataset shares several of its images with BV dataset.
    Process : all BV images are loaded first, then ET exclusive images are loaded.

    Params :
        csvBV, csvET : path of csv containing all filenames and corresponding labels
        BVpath, ETpath : path of directory from which images are loaded
    -----------
    Output :
        array : images from BV and ET dataset (resized)
        dataframe : image filenames with matching labels
    """

    resize_shape = (224,224)

    # get labels of both datasets as dataframes
    dfBV = pd.read_csv(csvBV,sep=';')
    dfET = pd.read_csv(csvET,sep=';')

    # get indexes of images that are exclusively in the ET dataset
    compare_ETidx = dfET['ImageID'].isin(dfBV['Original_img']) # True -> image is common to both series | False -> image is unique to ET series
    ETonly_indexes = compare_ETidx[compare_ETidx == False].index
    compare_BVidx = dfBV['Original_img'].isin(dfET['ImageID'])  # True -> image is common to both series | False -> image is unique to BV series
    BVonly_indexes = compare_BVidx[compare_BVidx == False].index

    # create new dataframe to store labels
    dfAll = dfBV.drop(columns=['New_name']) # re-use copy of dfBV
    dfAll = dfAll.rename(columns={"Original_img": "ImageID"})
    dfAll = dfAll.assign(Source='BV_ET') # add Source column to dfAll and assign BV_ET value to all rows
    dfAll = dfAll[['ImageID','GTExpression','GTEmotion','Source']]
    dfET = dfET.assign(Source='ET') # add Source column to dfET and assign ET value to all rows

    img_list = [] # images are temporarily stored in a list
    # load all BV images
    for i in range(dfBV.shape[0]) :
        if i in BVonly_indexes :
            dfAll.loc[i,'Source'] = 'BV' # update Source column in final dataframe
        # load BV image
        f = dfBV['New_name'][i]
        try:
            img = Image.open(os.path.join(imgsBV,f))
            img = img.resize(resize_shape) # resize
            img_list.append(np.array(img))
        except Exception as e:
            print(e)
    # load only ET exclusive images
    for i in ETonly_indexes :
        # insert row from ET dataframe into final dataframe
        dfAll = pd.concat([dfAll,dfET.loc[[i]]])
        # load ET image
        f = dfET['ImageID'][i]
        try:
            img = Image.open(os.path.join(imgsET,f))
            img = remove_pad_single(img) # padding needs to be removed from ET images
            img = img.resize(resize_shape) # resize
            img_list.append(np.array(img))
        except Exception as e:
            print(e)
    print("Loaded",dfAll.shape[0],"images successfully")

    dfAll = dfAll.reset_index(drop=True)
    print("Created dataframe for labels")

    return np.array(img_list), dfAll

## Counting human scores for each ET image

def human_scores_for_all(dfET,nb_people=50):

    """
    Load scores of participants for each ET image.

    Process : for each image, we store the number of votes in each category

    Params :
        dfET : path of csv containing all filenames and corresponding labels
        nb_people : total number of participants
    -----------
    Output :
        array : list of all scores in each category for each image

    """
    
    imgID_series = dfET['ImageID']
    list_all = []
    for i in range(imgID_series.size) :
        img_name = imgID_series[i] # get image filename
        list = [0,0,0,0]
        for k in range(nb_people):
            # open ET results for participant k+1
            if k<=8 :
                f = ".\\datasets\\labels\\ET_collected_labels\\et_ "+str(k+1)+".csv"
            elif k!=33 :
                f = ".\\datasets\\labels\\ET_collected_labels\\et_"+str(k+1)+".csv"
            df = pd.read_csv(f,sep=',',encoding='latin-1',header=None)
            # get emotion predicted by participant
            if k!=33 :
                emo = df.loc[df[0]==img_name][1].values[0]
                # increment score in corresponding category
                if emo=='Joie':
                    list[0]+=1
                elif emo=='Tristesse':
                    list[1]+=1
                elif emo=='Surprise':
                    list[2]+=1
                elif emo=='Colère':
                    list[3]+=1
        # get final result for ongoing image
        list_all.append(np.array(list))

    return np.array(list_all)

## Computing degree of agreement on highest scoring class

def highest_class_rate(array):
    max = np.max(array)
    total = np.sum(array)
    result = max/total
    return result*100

# def agreement_on_img(idx_in_imgs, scores):
#     array = scores[idx_in_imgs]
#     for i in range(1,array.shape[0]):
#         sum += array[i]*array[i-1]
#     result = sum/(np.sum(array)*(np.sum(array)-1))
#     return result*100

## Transforming images

def remove_pad_single(im,orig_size=(1080,1920),targetsize=(720,720)) :

    """
    Crop a given image, the resulting image is centered on the original one.
    The initial intention was to use this function to remove the padding of the Eye-Tracking images.

    Parameters :
        im : image to remove pad from (PIL type)
        originalsize : original size of image
        targetsize : desired size of cropped image
    ------------------------
    Output : cropped image (PIL type)
    """

    (origH,origW) = orig_size
    (targH,targW) = targetsize
    im = np.array(im)

    # compute position of the desired window
    min_y=(origH//2)-(targH//2)
    max_y=(origH//2)+(targH//2)
    min_x=(origW//2)-(targW//2)
    max_x=(origW//2)+(targW//2)
    # crop image
    crop = im[min_y:max_y, min_x:max_x]
    # convert to PIL Image type
    pil_crop = Image.fromarray(crop.astype('uint8'), 'RGB')

    return pil_crop

## Sorting images by a given characteristic to create multiple groups

# Sort images by label
def group_by_label(imgs, labels):
    column = "GTExpression"
    list = [],[],[],[]
    for i in range(labels.shape[0]) :
        match labels[column][i] :
            case 1 : list[0].append(imgs[i]) # add image to first list - happiness
            case 2 : list[1].append(imgs[i]) # add image to second list - sadness
            case 3 : list[2].append(imgs[i]) # add image to third list - surprise
            case 6 : list[3].append(imgs[i]) # add image to fourth list - anger 
    return list[0], list[1], list[2], list[3]

# Sort images by predicted emotion (human prediction)
def group_by_human_predict(imgs, labels):
    list = [],[],[],[]
    column = "Human_prediction"
    for i in range(labels.shape[0]) :
        if labels[column][i]>0:
            match labels[column][i] :
                case 1 : list[0].append(imgs[i]) # add image to first list - happiness
                case 2 : list[1].append(imgs[i]) # add image to second list - sadness
                case 3 : list[2].append(imgs[i]) # add image to third list - surprise
                case 6 : list[3].append(imgs[i]) # add image to fourth list - anger  
    return list[0], list[1], list[2], list[3]

# Get length of each list in a tuple
def get_distribution(tuple):
    list = [len(tuple[i]) for i in range(len(tuple))]
    return list
