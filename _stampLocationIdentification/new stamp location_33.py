# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:00:32 2020

@author: Henock
"""
 
import numpy as np
import cv2  
import imutils
import IMPM
 

# read image file 
frame = cv2.imread('../TrainTestData/Train/Normal/1 (17).jpg')  
frame_original=frame 

#resize image  500,353
img=frame
scale_percent = 500/img.shape[0] # percent of original size
width = int(img.shape[1] * scale_percent )
height = int(img.shape[0] * scale_percent )
dim = (width, height)

# resize image
frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  

img=frame
red_channel = cv2.cvtColor(img,cv2.COLOR_BGR2YUV) #[:,:,2]  #cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
# # # Blur the image
# blur = cv2.GaussianBlur(gray,(1,1),0)
# thresh = IMPM.thresholding(gray)

# define range of red color in HSV
lower_red = np.array([115, 94, 227])
upper_red = np.array([83, 114, 250])

thresh_inv = cv2.inRange (red_channel, lower_red, upper_red)
thresh_inv=cv2.blur(thresh_inv,(3,3))
ret,thresh_inv = cv2.threshold(thresh_inv,127,255,cv2.THRESH_BINARY)

# # Blur the image
blur = cv2.GaussianBlur(thresh_inv,(1,1),0)

thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]



# # Blur the image
# blur = cv2.GaussianBlur(thresh_inv,(1,1),0)

# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


# # display image
# cv2.imshow("img image", img)
# cv2.imshow("gray image", gray)
# cv2.imshow("thresh image", thresh)
# # cv2.imshow("opening image", opening)
# # cv2.imshow("canny image", canny)
# # Maintain output window until user presses a key
# cv2.waitKey(0)
# # Destroying present windows on screen
# cv2.destroyAllWindows()


thresh_inv = thresh #cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# # Convert BGR to HSV
# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# # define range of red color in HSV
# lower_red = np.array([0, 78, 100])
# upper_red = np.array([346, 255, 255])

# thresh_inv = cv2.inRange (hsv, lower_red, upper_red)
# thresh_inv=cv2.blur(thresh_inv,(3,3))
# ret,thresh_inv = cv2.threshold(thresh_inv,127,255,cv2.THRESH_BINARY)


# # Blur the image
# blur = cv2.GaussianBlur(thresh_inv,(1,1),0)

# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

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

cv2.imshow("frame", frame)
cv2.imshow("boxes", mask)
cv2.imshow("final image", res_final)
cv2.waitKey(0)
cv2.destroyAllWindows()









