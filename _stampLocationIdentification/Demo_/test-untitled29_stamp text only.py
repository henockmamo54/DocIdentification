# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 20:36:23 2021

@author: Henock
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:26:51 2020

@author: Henock
"""
 
import cv2 
import numpy as np 
import pytesseract 
import pandas as pd
import re

stamps=[]
stringmatches=False

image = cv2.imread('image/normal/1 (11).jpg')  
frame_original=image  
 
scale_percent = 500/image.shape[0] # percent of original size
width = int(image.shape[1] * scale_percent )
height = int(image.shape[0] * scale_percent )
dim = (width, height)
# resize image
image= cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # define range of red color in HSV
thresh_inv1 = cv2.inRange (hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
thresh_inv2 = cv2.inRange (hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))    
thresh_inv= thresh_inv1 | thresh_inv2

# Blur the image
thresh=thresh_inv= cv2.blur(thresh_inv,(3,3)) 


#==================================================================================
#==================================================================================
#====================  Find the location of the stamp =============================
#==================================================================================
#==================================================================================

# find contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
mask = np.ones(image.shape[:2], dtype="uint8") * 255


image_withstampLocator = frame_original;

for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    if w*h>1000:
    # if w>width*0.6 and h<10 and h>1 :
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -2)
        image_withstampLocator=cv2.rectangle(image_withstampLocator, (int(x/scale_percent), int(y/scale_percent)), 
                      
                      (int((x+w)/scale_percent), int( (y+h)/scale_percent ) ), (120, 255, 0), 10)
        print(w*h)
        
        crop_img = image[y:y+h, 0:x+w]     
        cv2.imshow("cropped", crop_img)
        
        crop_img = frame_original[int(y/scale_percent):int((y+h)/scale_percent), 0:]           
        # crop_img = image[y:y+h+10, 0:]  
        
        # cv2.imshow("cropped", crop_img) 
        cv2.imwrite('crop_img.png',crop_img)
        
        cv2.waitKey(0)
        
        stamps.append(crop_img)

print ("******************************************************* \n      Stamp count = ",len(stamps))



# res_final = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)) 
# cv2.imshow("image", image)
# cv2.imshow("boxes", mask)
# cv2.imshow("final image", res_final)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


if(len(stamps)==1):
        
    
    #==================================================================================
    #==================================================================================
    #=================  Read the Text inside the cropped image ========================
    #==================================================================================
    #==================================================================================
    
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    crop_img=cv2.inRange (crop_img, np.array([0, 0, 0]), np.array([180, 255, 30]))
    rect,crop_img = cv2.threshold(crop_img,0,255,cv2.THRESH_BINARY )
    crop_img = cv2.erode(crop_img, np.ones((5,5),np.uint8), iterations = 1)    
    kernel = np.ones((7,7),np.uint8)
    crop_img = cv2.dilate(crop_img, kernel, iterations = 1)
     
    width = int(crop_img.shape[1] *3* scale_percent )
    height = int(crop_img.shape[0] *3* scale_percent )
    dim = (width, height)
    cv2.imshow("captured text", cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
     
    
    #configuring parameters for tesseract
    custom_config = r'--oem 3 --psm 6  ' #-ctessedit_char_blacklist= 0123456789
    
    # chekc the text next to stamp
    text = pytesseract.image_to_string(crop_img, lang = 'kor', config=custom_config).replace('\n','').strip() 
    text=re.sub('[A-Za-z0-9]+', '', text) 
    
    
    
    # now feeding image to tesseract
    details = pytesseract.image_to_data(crop_img, output_type=pytesseract.Output.DICT, config=custom_config, lang="kor")
    temp=pd.DataFrame(details)
    temp.conf=pd.to_numeric(temp.conf)
    temp= temp[temp.conf>30]

    crop_img = cv2.rectangle(crop_img, (np.min(temp.left),np.min(temp.top) ), 
                             ( np.max(temp.left)+np.max(temp.width)   , np.max(temp.top)+np.max(temp.height)), (255,255,255), 10)
        
    
    cv2.imwrite('crop_img2.png',crop_img)
            
    width = int(crop_img.shape[1] * scale_percent )
    height = int(crop_img.shape[0] * scale_percent )
    dim = (width, height)
    
    cv2.imshow("captured text222", cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    