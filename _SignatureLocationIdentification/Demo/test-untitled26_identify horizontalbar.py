# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:53:33 2021

@author: Henock
"""

 
import cv2 
import numpy as np 
import pytesseract 
import pandas as pd
import re
  
image = cv2.imread('image/normal/1 (43).jpg') 
frame_original=image  
 
# resize image
scale_percent = 900/image.shape[0] # percent of original size
width = int(image.shape[1] * scale_percent )
height = int(image.shape[0] * scale_percent )
dim = (width, height)
image= cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
 


# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # define range color in HSV
thresh_inv = cv2.inRange (hsv, np.array([0, 0, 200]), np.array([0, 0, 204])) 
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

horizontalbars=[]

for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # print(w*h)
    if w*h>2000:
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
        
        horizontalbars.append([crop_img, int(y/scale_percent)])







