# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:15:30 2021

@author: Henock
"""


import cv2 
import numpy as np 
import pytesseract 
import pandas as pd
import re
from fuzzywuzzy import fuzz
from os import walk 


_, _, filenames = next(walk("image/normal"))

for file in filenames:  
    
    print(file)
        
    stamps=[]
    stringmatches=False
    
    image = cv2.imread('image/normal/'+file)  
    frame_original=image  
    image_withstampLocator = image;
     
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
    
    image_withstampLocator = frame_original;
    
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if w*h>1000: 
            
            image_withstampLocator=cv2.rectangle(image_withstampLocator, (int(x/scale_percent), int(y/scale_percent)),
                                                 (int((x+w)/scale_percent), int( (y+h)/scale_percent ) ), (140, 255, 0), 15) 
            image_withstampLocator=cv2.rectangle(image_withstampLocator, (0+int(10/scale_percent), int(y/scale_percent)), 
                                                 (int((x+w)/scale_percent), int( (y+h)/scale_percent ) ), (120, 100, 0), 10)
            
            # crop_img = image[y:y+h, 0:x+w]     
            # cv2.imshow("cropped", crop_img)
            
            crop_img = frame_original[int(y/scale_percent):int((y+h)/scale_percent), 0:]           
             
            # cv2.imwrite('crop_img.png',crop_img)
             
            
            stamps.append([crop_img,int(y/scale_percent)])
    
    print ("******************************************************* \n      Stamp count = ",len(stamps))
    
    
    
    
    
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
        crop_img=255-crop_img
         
        # width = int(crop_img.shape[1] *3* scale_percent )
        # height = int(crop_img.shape[0] *3* scale_percent )
        # dim = (width, height)
        # cv2.imshow("captured text", cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
         
        
        #configuring parameters for tesseract
        custom_config = r'--oem 3 --psm 6  ' #-ctessedit_char_blacklist= 0123456789
        
        # chekc the text next to stamp
        text = pytesseract.image_to_string(crop_img, lang = 'kor', config=custom_config).replace('\n','').strip() 
        text=re.sub('[A-Za-z0-9]+', '', text) 
        stamptext=text.strip().replace("\n","").replace(" ","")
        
        
        
        # now feeding image to tesseract
        details = pytesseract.image_to_data(crop_img, output_type=pytesseract.Output.DICT, config=custom_config, lang="kor")
        temp=pd.DataFrame(details)
        temp.conf=pd.to_numeric(temp.conf)
        temp= temp[temp.conf>30]
    
        crop_img = cv2.rectangle(crop_img, (np.min(temp.left),np.min(temp.top) ), 
                                 ( np.max(temp.left)+np.max(temp.width)   , np.max(temp.top)+np.max(temp.height)), (255,255,255), 10)
        
        image_withstampLocator = cv2.rectangle(image_withstampLocator, (np.min(temp.left),stamps[0][1]+np.min(temp.top) ), 
                                 ( np.max(temp.left)+np.max(temp.width)   , stamps[0][1]+np.max(temp.top)+np.max(temp.height)), (115, 24, 165), 20)
            
        
        # cv2.imwrite('crop_img2.png',crop_img)
                
        # width = int(crop_img.shape[1] * scale_percent )
        # height = int(crop_img.shape[0] * scale_percent )
        # dim = (width, height)
        
        # cv2.imshow("captured text222", cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        # cv2.imwrite('image_withstampLocator.png',image_withstampLocator)
         
        
        
        #==================================================================================
        #==================================================================================
        #=================  Read the header text from original image ========================
        #==================================================================================
        #==================================================================================
         
        # header_image1 = frame_original[0:frame_original.shape[1],0:int(frame_original.shape[0]/4),]
        header_image = frame_original[0:int(frame_original.shape[0]/5.5),:,]
        
        
        header_image=cv2.cvtColor(header_image,cv2.COLOR_BGR2GRAY)
        rect,header_image = cv2.threshold(header_image,0,255,cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        header_image=cv2.morphologyEx(header_image, cv2.MORPH_OPEN, kernel)
        header_image = cv2.dilate(header_image, np.ones((5,5),np.uint8), iterations = 1)
        rect,header_image = cv2.threshold(header_image,0,255,cv2.THRESH_BINARY)
        header_image = cv2.erode(header_image, np.ones((9,9),np.uint8), iterations = 1)
        
            
        
        details = pytesseract.image_to_data(header_image, output_type=pytesseract.Output.DICT, config=custom_config, lang="kor")
        temp= pd.DataFrame(details)
        
        temp=temp.query(" (line_num <= 2)  ").reset_index()
        temp["test_len"] = 0
        for i in range(temp.shape[0]):
            # temp.iloc[i,-1]=len(temp.text[i])  
            temp.iloc[i,-1]=len(''.join([i for i in temp.text[i].strip() if  i.isalpha()]))    
            
        temp= temp[temp.test_len>0]  
        temp= temp[temp.conf>30]
        temp= temp[temp.height>100]
        
        header_text="".join(temp.text).strip()
        header_text=re.sub('[A-Za-z0-9]+', '', header_text).replace("\n","").replace(" ","")
        
        
        padding=50
            
        header_image = cv2.rectangle(header_image, (np.min(temp.left),np.min(temp.top) ), 
                                     ( np.max(temp.left)+np.max(temp.width)   , np.max(temp.top)+np.max(temp.height)), (120, 255, 0), 10)
        
        image_withstampLocator=cv2.rectangle(image_withstampLocator, (np.min(temp.left),np.min(temp.top) ), 
                                      ( np.max(temp.left)+np.max(temp.width)   , np.max(temp.top)+np.max(temp.height)), (115, 24, 165), 20)
        
    
        # cv2.imwrite('image_withstampLocator.png',image_withstampLocator)
        
        
        
        #==================================================================================
        #==================================================================================
        #============  compare the text next to the stamp and header text =================
        #==================================================================================
        #==================================================================================
         
        
        print(header_text, "\n" ,stamptext)
        print(fuzz.token_set_ratio(header_text, stamptext))
        
        stringmatches= (fuzz.token_set_ratio(header_text, stamptext)>0)
        
    #==================================================================================
    #==================================================================================
    
    
    if(len(stamps)>0 and stringmatches):
        print("--------------> normal")
        cv2.imwrite('result/normal/normal/'+file,image_withstampLocator)
    else:
        print("--------------> not normal")        
        cv2.imwrite('result/normal/notnormal/'+file,image_withstampLocator)
        
        