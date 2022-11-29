from deepface import DeepFace
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from PIL import Image
import os, os.path

# Load all images from the BV dataset
def load_images() :
    img_list = []
    path = ".\datasets\BV"
    for f in os.listdir(path):
        try:
            img = Image.open(os.path.join(path,f))
            # img = np.array(img)
            img_list.append(img)
        except Exception as e:
            print(e)
    return np.array(img_list)

# Display given number of RGB images from an array, starting from a given position in the array
def disp_ims(imgs, nb_imgs=10, start_index=0) :
    plt.figure(figsize=(10,10))
    for i in range(nb_imgs) :
        plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[start_index+i])
        # plt.title(image)
        plt.axis("off")

# Display grayscale image
def gr_show_im(gray_im) :
    arr = np.array(gray_im)
    plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
    plt.show()

def preproc_deepface (img) :
    target_size=(48,48)
    grayscale = True
    enforce_detection = False
    detector_backend = 'opencv'
    return_region = False
    result = DeepFace.functions.preprocess_face(img = img, target_size = target_size, grayscale = grayscale, enforce_detection = enforce_detection, detector_backend = detector_backend, return_region = return_region)

    return result

def preproc_fer (img_array) :
    result = []
    for img in img_array :
        img = np.array(img)
        proc_img = preproc_deepface(img)
        result.append(proc_img)
    result = np.array(result)
    return result