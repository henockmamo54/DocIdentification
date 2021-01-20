# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:57:17 2020

@author: Henock
"""
  
 
import numpy as np
import cv2  
import imutils
 
# frame = cv2.imread('TestImages/ct2.png') 
frame = cv2.imread('../TrainTestData/Train/Normal/1 (59).jpg')   

#resize image
img=frame
scale_percent = 500/img.shape[0] # percent of original size
width = int(img.shape[1] * scale_percent )
height = int(img.shape[0] * scale_percent )
dim = (width, height)

# resize image
frame_original = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
frame=frame_original

# kernel = np.ones((5,5),np.float32)/25
# # frame = cv2.filter2D(frame_original,-1,kernel)
# frame=cv2.blur(frame_original,(2,2))

# crop_img = frame[int(height*0.5):height, 0:width]     
# cv2.imshow("cropped", crop_img)

# cv2.imshow("frame_orig╧inal",frame_original)
# cv2.imshow("frame",frame)
# cv2.waitKey(0)


img=frame
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

thresh_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# Blur the image
blur = cv2.GaussianBlur(thresh_inv,(1,1),0)

thresh = cv2.threshold(blur, 0, 235, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# find contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

mask = np.ones(img.shape[:2], dtype="uint8") * 255
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # if w*h>1000:
    if w>width*0.6 and h<5 and h>1 :
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -1)
        print(w*h)

res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

cv2.imshow("gray", gray)
cv2.imshow("frame", frame)
cv2.imshow("boxes", mask)
cv2.imshow("final image", res_final)
cv2.waitKey(0)
cv2.destroyAllWindows()






 

# # Convert BGR to HSV
# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# # define range of red color in HSV
# lower_red = np.array([0, 78, 100])
# upper_red = np.array([346, 255, 255])

# mask = cv2.inRange (hsv, lower_red, upper_red)
# mask=cv2.blur(mask,(3,3))
# ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# mask=cv2.dilate(mask,kernel,iterations = 1)
# mask=cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)


# contours, _ = cv2.findContours(mask.copy(),
#                             cv2.RETR_TREE,
#                             cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True) 
#     if len(approx)>3:
#         cv2.drawContours(frame,[cnt],0,255,-1)   
#         cv2.drawContours(mask,[cnt],0,255,-2)   
        
        
# mask=cv2.blur(mask,(3,3)) 
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# mask=cv2.dilate(mask,kernel,iterations = 1)  

# mask=cv2.blur(mask,(3,3)) 
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# mask=cv2.dilate(mask,kernel,iterations = 1)  
    
        
# contours, _ = cv2.findContours(mask.copy(),
#                             cv2.RETR_TREE,
#                             cv2.CHAIN_APPROX_SIMPLE) 


# cv2.imshow('mask',mask)
# cv2.imshow('frame', frame)
# cv2.waitKey(0)
        
        


# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#     print (len(approx))
#     if len(approx)>3: 
#         cv2.drawContours(frame,[cnt],0,255,-1)   
#         cv2.drawContours(mask,[cnt],0,255,-2)    

# cv2.imshow('mask',mask)
# cv2.imshow('frame', frame)
# cv2.waitKey(0)



# # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
# #  	cv2.CHAIN_APPROX_SIMPLE)
# # cnts = imutils.grab_contours(cnts)
# # print("I found {} black shapes".format(len(cnts)))
# # cv2.imshow("Mask", mask)
# # cv2.waitKey(0)

# # # loop over the contours
# # for c in cnts:
# #  	# draw the contour and show it
# #  	cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
# #  	cv2.imshow("Image", frame)
# #  	cv2.waitKey(0)
    
    
    

# # cropedimages=[]

# # if len(contours) > 0:
# #     red_area = max(contours, key=cv2.contourArea)
# #     x, y, w, h = cv2.boundingRect(red_area)
# #     cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 0, 255), 2)
    
# #     cv2.rectangle(frame,(0, y),(x+w, y+h+10),(0, 255, 0), 2)

# #     crop_img = frame[y:y+h, 0:x+w]     
# #     cv2.imshow("cropped", crop_img)
# #     cropedimages.append(crop_img)
    
# #     print((x, y),(x+w, y+h))


# # cv2.imshow('frame', frame)
# # cv2.imshow('mask', mask)
# # cv2.imshow("cropped", crop_img)

# # cv2.waitKey(0)


# # import pytesseract 

# # print(pytesseract.image_to_string(crop_img))




