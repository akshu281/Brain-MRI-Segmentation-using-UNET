from skimage.measure import compare_ssim
import imutils
import cv2
import time
import numpy as np
import math

def dice():
    for i in range(0,6):
        im1=cv2.imread("data/brain/test/"+str(i)+"_predict.png")
        im1=cv2.resize(im1,(256,215))
        im2=cv2.imread("data/brain/manual_masks/"+str(i)+".png")
        intersection=np.bitwise_and(im2,im1)
        score=2*np.sum(intersection)/(np.sum(im1)+np.sum(im2))
        print("Dice_Test"+str(i), score)

    for i in range(0,6):
        im1=cv2.imread("data/brain/baseline_masks/"+str(i)+".png")
        im2=cv2.imread("data/brain/manual_masks/"+str(i)+".png")
        intersection=np.bitwise_and(im2,im1)
        score=2*np.sum(intersection)/(np.sum(im1)+np.sum(im2))
        print("Dice_Baseline"+str(i), score)

dice()