# from deepface import DeepFace
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

def loadET(csv,path) :

    """
    Load all Eye-Tracking images using filenames indicated in csv

    Parameter :
        my_csv : path of the csv file containing all image filenames with corresponidng labels
        my_path : path of the directory to load all images from
    ------------------------
    Output : array containing loaded images (in PIL type)
    """

    img_list = []
    # read csv
    df_labels = pd.read_csv(csv,sep=';')
    # read filename from first column of dataframe and load corresponding image
    for i in range(df_labels['ImageID'].size):
        f = df_labels['ImageID'][i]
        try:
            img = Image.open(os.path.join(path,f))
            img = remove_pad_single(img)
            img_list.append(img)
        except Exception as e:
            print(e)
    return np.array(img_list)

def loadBV(csv,path) :

    """
    Load all Bubble View images using filenames indicated in csv

    Parameter :
        my_csv : path of the csv file containing all image filenames with corresponidng labels
        my_path : path of the directory to load all images from
    ------------------------
    Output : array containing loaded images (in PIL type)
    """

    img_list = []
    # read csv
    df_labels = pd.read_csv(csv,sep=';')
    # read filename from first column of dataframe and load corresponding image
    for i in range(df_labels['New_name'].size):
        f = df_labels['New_name'][i]
        try:
            img = Image.open(os.path.join(path,f))
            img_list.append(img)
        except Exception as e:
            print(e)
    return np.array(img_list)

def load_all(csvBV, BVimgs_path, csvET, ETimgs_path):

    """
    Load all Bubble View and Eye Tracking images using filenames indicated in csv.
    Note that ET dataset shares several of its images with BV dataset. Thus all BV images are loaded, then relevant ET images are sorted then loaded.

    Params :
        csvBV, csvET : path of BV and ET csv containing all filenames and corresponding labels
        BVpath, ETpath : path of directory from which images are loaded
    -----------
    Output :
        array containing images from BV and ET dataset (images of PIL type)
        and dataframe containing all matching labels
    """

    resize_shape = (224,224)

    # get labels of both datasets
    dfBV = pd.read_csv(csvBV,sep=';')
    dfET = pd.read_csv(csvET,sep=';')
    # get indexes of ET images that were not also used in BV dataset
    compare_ETidx = dfET['ImageID'].isin(dfBV['Original_img']) # True -> image is common to both series | False -> image is unique to ET series
    ETonly_imgs_index = compare_ETidx[compare_ETidx == False].index
    compare_BVidx = dfBV['Original_img'].isin(dfET['ImageID'])  # True -> image is common to both series | False -> image is unique to BV series
    BVonly_imgs_index = compare_BVidx[compare_BVidx == False].index
    # print("Detected",ETonly_imgs_index.shape[0],"relevant images in ET dataset")

    # create new dataframe to store labels
    dfAll = dfBV.drop(columns=['New_name']) # re-use copy of dfBV
    dfAll = dfAll.rename(columns={"Original_img": "ImageID"})
    dfAll = dfAll.assign(Source='BV_ET') # add Source column to dfAll and assign BV_ET value to all rows
    dfAll = dfAll[['ImageID','GTExpression','GTEmotion','Source']]
    dfET = dfET.assign(Source='ET') # add Source column to dfET and assign ET value to all rows

    img_list = [] # images are temporarily stored in a list
    # load all BV images
    for i in range(dfBV['New_name'].size) :
        # update Source column in final dataframe
        if i in BVonly_imgs_index :
            dfAll.loc[i,'Source'] = 'BV'
        # load BV image
        f = dfBV['New_name'][i]
        try:
            img = Image.open(os.path.join(BVimgs_path,f))
            img = img.resize(resize_shape)
            img_list.append(np.array(img))
        except Exception as e:
            print(e)
    # load only relevant ET images
    for i in ETonly_imgs_index :
        # insert row from ET dataframe into final dataframe
        dfAll = pd.concat([dfAll,dfET.loc[[i]]])
        # load ET image
        f = dfET['ImageID'][i]
        try:
            img = Image.open(os.path.join(ETimgs_path,f))
            img = remove_pad_single(img) # padding needs to be removed from ET images
            img = img.resize(resize_shape)
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
    Pour chaque image :
        Créer liste de taille 4
        Pour chaque personne :
            Trouver image dans dataframe :
                Stocker emotion => +1 dans la liste à l'indice correspondant à emotion
    """
    
    imgID_series = dfET['ImageID']
    list_all = []
    for i in range(imgID_series.size) :
        img_name = imgID_series[i]
        list = [0,0,0,0]
        for k in range(nb_people):
            # open ET results for participant k+1
            if k<=8 :
                f = ".\\datasets\\ET_collected_labels\\et_ "+str(k+1)+".csv"
            elif k!=33 :
                f = ".\\datasets\\ET_collected_labels\\et_"+str(k+1)+".csv"
            df = pd.read_csv(f,sep=',',encoding='latin-1',header=None)
            # get emotion predicted by partipant
            if k!=33 :
                emo = df.loc[df[0]==img_name][1].values[0]
                if emo=='Joie':
                    list[0]+=1
                elif emo=='Tristesse':
                    list[1]+=1
                elif emo=='Surprise':
                    list[2]+=1
                elif emo=='Colère':
                    list[3]+=1
        list_all.append(np.array(list))
        # print(i+1,img_name,list)
    return np.array(list_all)

## Computing percentage of highest value in a given array

def mean_score(array):
    maxi = np.max(array)
    total = np.sum(array)
    result = maxi/total
    return result*100

def agreement_on_img(idx_in_imgs, scores):
    array = scores[idx_in_imgs]
    for i in range(1,array.shape[0]):
        sum += array[i]*array[i-1]
    result = sum/(np.sum(array)*(np.sum(array)-1))
    return result*100

## Transform images

def remove_pad_single(im,originalsize=(1080,1920),targetsize=(720,720)) :

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

    (origH,origW) = originalsize
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

def remove_pad_all(im_arr,originalsize=(1080,1920),targetsize=(720,720)) :

    """
    Crop images, each resulting image is centered on the original one.
    The initial intention was to use this function to remove the padding of the Eye-Tracking images

    Parameters :
        im_arr : array of images (PIL type)
        originalsize : original size of images
        targetsize : desired size of cropped images
    ------------------------
    Output : array of cropped images (PIL type)
    """

    (origH,origW) = originalsize
    (targH,targW) = targetsize
    img_list = [] # cropped images are temporarily stored in a list

    for image in im_arr :
        im = np.array(image)
        # compute position of the desired window
        min_y=(origH//2)-(targH//2)
        max_y=(origH//2)+(targH//2)
        min_x=(origW//2)-(targW//2)
        max_x=(origW//2)+(targW//2)
        # crop image
        crop = im[min_y:max_y, min_x:max_x]
        # convert to PIL Image type
        pil_crop = Image.fromarray(crop.astype('uint8'), 'RGB')
        img_list.append(pil_crop)

    return np.array(img_list)

# def preproc_deepface (img) :
#     target_size=(48,48)
#     grayscale = True
#     enforce_detection = False
#     detector_backend = 'opencv'
#     return_region = False
#     result = DeepFace.functions.preprocess_face(img = img, target_size = target_size,
#         grayscale = grayscale, enforce_detection = enforce_detection,
#         detector_backend = detector_backend, return_region = return_region)
#     return result

# def preproc_fer (img_array) :
#     result = []
#     for img in img_array :
#         img = np.array(img)
#         proc_img = preproc_deepface(img)
#         result.append(proc_img)
#     result = np.array(result)
#     return result