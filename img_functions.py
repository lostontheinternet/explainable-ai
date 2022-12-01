from deepface import DeepFace

import random
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
from PIL import Image
from torchvision import transforms

def load_ims(path) :

    """
    Load all images from a given path
    Parameter : path of the directory from which to load all images

    Output : array containing loaded images (in PIL type)
    """

    img_list = [] # loaded images are temporarily stored in a list
    for f in os.listdir(path):
        try:
            img = Image.open(os.path.join(path,f))
            # img = np.array(img)
            img_list.append(img)
        except Exception as e:
            print(e)
    return np.array(img_list)

def resize_ims(im_arr,targetsize) :

    """
    Resize all images from a given numpy array
    Parameters :
        im_arr : array of images (type PIL)
        sizeH, sizeW : dimensions to resize images to
    
    Output : array of resized images (type PIL)
    """

    H,W = targetsize
    result=[] # resized images are temporarily stored in a list
    resize=transforms.Resize(size=(H,W))
    for img in im_arr:
        img = resize(img)
        result.append(img)
    return np.array(result)

def crop_ims(im_arr,originalsize,targetsize):

    """
    Crop images, each resulting image is centered on the original one
    The initial intention was to use this function to remove the padding of the Eye-Tracking images
    Parameters :
        im_arr : array of images (PIL type)
        originalsize : original size of images
        targetsize : desired size of cropped images

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

def preproc_deepface (img) :
    target_size=(48,48)
    grayscale = True
    enforce_detection = False
    detector_backend = 'opencv'
    return_region = False
    result = DeepFace.functions.preprocess_face(img = img, target_size = target_size,
        grayscale = grayscale, enforce_detection = enforce_detection,
        detector_backend = detector_backend, return_region = return_region)
    return result

def preproc_fer (img_array) :
    result = []
    for img in img_array :
        img = np.array(img)
        proc_img = preproc_deepface(img)
        result.append(proc_img)
    result = np.array(result)
    return result