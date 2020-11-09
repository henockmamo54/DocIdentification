# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:57:17 2020

@author: Henock
"""
  
 
import numpy as np
import cv2  
import imutils
 
# frame = cv2.imread('TestImages/ct2.png') 
frame = cv2.imread('../TrainTestData/Train/NotNormal/46 (29).jpg')   

#resize image  500,353
img=frame
scale_percent = 500/img.shape[0] # percent of original size
# width = int(img.shape[1] * scale_percent )
# height = int(img.shape[0] * scale_percent )
width = 353
height = 500
dim = (width, height)

# resize image
frame_original = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
frame=frame_original



img=frame
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# Blur the image
blur = cv2.GaussianBlur(thresh_inv,(1,1),0)

thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# find contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

mask = np.ones(img.shape[:2], dtype="uint8") * 255
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # if w*h>1000:
    if w>width*0.6 and h<10 and h>1 :
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -2)
        print(w*h)

res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

cv2.imshow("gray", gray)
cv2.imshow("frame", frame)
cv2.imshow("boxes", mask)
cv2.imshow("final image", res_final)
cv2.waitKey(0)
cv2.destroyAllWindows()



