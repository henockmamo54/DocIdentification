# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 00:05:35 2020

@author: Henock
"""
 
import numpy as np
import cv2  

 
frame = cv2.imread('TestImages/ct2.png')
# width=frame.size[0]

# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of red color in HSV
lower_red = np.array([0, 78, 100])
upper_red = np.array([346, 255, 255])

mask = cv2.inRange (hsv, lower_red, upper_red)
contours, _ = cv2.findContours(mask.copy(),
                           cv2.RETR_TREE,
                           cv2.CHAIN_APPROX_SIMPLE)

cropedimages=[]

if len(contours) > 0:
    red_area = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(red_area)
    cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 0, 255), 2)
    
    cv2.rectangle(frame,(0, y),(x+w, y+h+10),(0, 255, 0), 2)

    crop_img = frame[y:y+h, 0:x+w]     
    cv2.imshow("cropped", crop_img)
    cropedimages.append(crop_img)
    
    print((x, y),(x+w, y+h))
""

cv2.imshow('frame', frame)
cv2.imshow('mask', mask)
cv2.imshow("cropped", crop_img)

cv2.waitKey(0)


import pytesseract 

print(pytesseract.image_to_string(crop_img))
