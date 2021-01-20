# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 14:43:41 2021

@author: Henock
"""


 
import cv2 
import numpy as np 
import pytesseract 
import pandas as pd
import re

# image = cv2.imread('image/normal/1 (1).jpg')  
image = cv2.imread('image/normal/1 (72).jpg') 
frame_original=image  
 
scale_percent = 900/image.shape[0] # percent of original size
width = int(image.shape[1] * scale_percent )
height = int(image.shape[0] * scale_percent )
dim = (width, height)
# resize image
image= cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
 

# show annotated image and wait for keypress
cv2.imshow("crop_img2", cv2.resize(image, (width,height), interpolation = cv2.INTER_AREA) )
cv2.waitKey(0)
cv2.destroyAllWindows() 


# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # define range of red color in HSV
thresh_inv1 = cv2.inRange (hsv, np.array([0, 0, 200]), np.array([0, 0, 204]))
# thresh_inv2 = cv2.inRange (hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))    
# thresh_inv= thresh_inv1 | thresh_inv2
thresh_inv=thresh_inv1

# Blur the image
thresh=thresh_inv= cv2.blur(thresh_inv,(3,3)) 


# show annotated image and wait for keypress
cv2.imshow("crop_img2", cv2.resize(thresh, (width,height), interpolation = cv2.INTER_AREA) )
cv2.waitKey(0)
cv2.destroyAllWindows() 

#==================================================================================
#==================================================================================
#====================  Find the location of the stamp =============================
#==================================================================================
#==================================================================================

# find contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
mask = np.ones(image.shape[:2], dtype="uint8") * 255

stamps=[]

for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # print(w*h)
    if w*h>1000:
    # if w>width*0.6 and h<10 and h>1 :
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -2)
        print(w*h)
        
        crop_img = image[y:y+h, 0:x+w]     
        cv2.imshow("cropped", crop_img)
        
        crop_img = frame_original[int(y/scale_percent):int((y+h+5)/scale_percent), 0:]           
        # crop_img = image[y:y+h+10, 0:]  
        
        # cv2.imshow("cropped", crop_img) 
        cv2.imwrite('crop_img.png',crop_img)
        
        cv2.waitKey(0)
        
        stamps.append(crop_img)





