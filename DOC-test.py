# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 23:39:14 2020

@author: Henock
"""
import numpy as np
import cv2  

frame = cv2.imread('ct.png')
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of red color in HSV
lower_red = np.array([0, 78, 100])
upper_red = np.array([341, 255, 255])

mask = cv2.inRange (hsv, lower_red, upper_red)
contours, _ = cv2.findContours(mask.copy(),
                           cv2.RETR_TREE,
                           cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    red_area = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(red_area)
    cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 0, 255), 2)
    
    cv2.rectangle(frame,(0, y),(x+w, y+h+10),(0, 255, 0), 2)
    
    print((x, y),(x+w, y+h))
""

cv2.imshow('frame', frame)
cv2.imshow('mask', mask)

cv2.waitKey(0)