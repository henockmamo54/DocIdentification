# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:22:29 2021

@author: Henock
"""




import cv2  
import re
import imutils
import numpy as np
import pytesseract  
from matplotlib import pyplot as plt

 
# frame = cv2.imread('TestImages/ct2.png') 
frame = cv2.imread('../TrainTestData/Train/NotNormal/46 (8).jpg')  
# frame=cv2.imread("sample.jpg")
frame_original=frame 

#resize image  500,353
img=frame
scale_percent = 500/img.shape[0] # percent of original size
width = int(img.shape[1] * scale_percent )
height = int(img.shape[0] * scale_percent )
# width = 353
# height = 500
dim = (width, height)

# resize image
frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 



img=frame
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
# Blur the image
blur = cv2.GaussianBlur(thresh,(1,1),0)

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
        
        crop_img = frame[y+1:, 0:] 
        # cv2.imshow("cropped", crop_img) 
        
        crop_img = frame_original[int((y+1)/scale_percent):, 0:]   
        cv2.imwrite('crop_img.png',crop_img)


crop_img=cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
rect,crop_img = cv2.threshold(crop_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
crop_img = cv2.erode(crop_img, kernel, iterations = 1)
crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel)



# cv2.imshow("crop_img", crop_img) 
# cv2.waitKey(0)
# cv2.destroyAllWindows() 




#configuring parameters for tesseract
custom_config = r'--oem 3 --psm 6  ' #-ctessedit_char_blacklist= 0123456789

# chekc the text next to stamp
text = pytesseract.image_to_string(crop_img, lang = 'kor', config=custom_config).strip() 
boxes =pytesseract.image_to_boxes(crop_img, lang = 'kor', config=custom_config)




  
# Finding contours 
contours, hierarchy = cv2.findContours(crop_img, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE) 
  
# Creating a copy of image 
im2 = img.copy() 
  
# A text file is created and flushed 
file = open("recognized.txt", "w+") 
file.write("") 
file.close() 
  
# Looping through the identified contours 
# Then rectangular part is cropped and passed on 
# to pytesseract for extracting text from it 
# Extracted text is then written into the text file 
for cnt in contours: 
    x, y, w, h = cv2.boundingRect(cnt) 
      
    # Drawing a rectangle on copied image 
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
      
    # Cropping the text block for giving input to OCR 
    cropped = im2[y:y + h, x:x + w] 
      
    # Open the file in append mode 
    file = open("recognized.txt", "a") 
      
    # Apply OCR on the cropped image 
    text = pytesseract.image_to_string(cropped) 
      
    # Appending the text into file 
    file.write(text) 
    file.write("\n") 
      
    # Close the file 
    file.close 











