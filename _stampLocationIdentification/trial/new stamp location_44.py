# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:20:51 2020

@author: Henock
""" 

import numpy as np
import cv2  
import imutils
 

frame = cv2.imread('../TrainTestData/Train/Normal/1 (19).jpg')  
frame_original=frame 


img=frame
scale_percent = 500/img.shape[0] # percent of original size
width = int(img.shape[1] * scale_percent )
height = int(img.shape[0] * scale_percent )
dim = (width, height)
# resize image
img=frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of red color in HSV
lower_red = np.array([0, 20, 100])
upper_red = np.array([346, 255, 255])

thresh_inv = cv2.inRange (hsv, lower_red, upper_red)

# Blur the image
thresh=thresh_inv= cv2.blur(thresh_inv,(3,3)) 



# find contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
mask = np.ones(img.shape[:2], dtype="uint8") * 255

for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    if w*h>1000:
    # if w>width*0.6 and h<10 and h>1 :
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -2)
        print(w*h)
        
        crop_img = frame[y:y+h, 0:x+w]     
        cv2.imshow("cropped", crop_img)
        
        crop_img = frame_original[int(y/scale_percent):int((y+h+5)/scale_percent), 0:]     
        # cv2.imshow("cropped", crop_img) 
        cv2.imwrite('crop_img.png',crop_img)
        
        cv2.waitKey(0)


res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

# cv2.imshow("gray", gray)
cv2.imshow("frame", frame)
cv2.imshow("boxes", mask)
cv2.imshow("final image", res_final)
cv2.waitKey(0)
cv2.destroyAllWindows()






