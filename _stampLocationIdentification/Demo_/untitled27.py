# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:22:30 2021

@author: Henock
"""

import cv2    
import numpy as np
import pandas as pd
import pytesseract   
from scipy.spatial import distance 
from os import walk
import re


_, _, filenames = next(walk("image/normal"))

for file in filenames:        
    print(file)  
    
    stamps=[]
    stringmatches=False
    
    image = cv2.imread('image/normal/'+file)  
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
            image_withstampLocator=cv2.rectangle(image_withstampLocator, (0+int(10/scale_percent), int(y/scale_percent)), 
                          
                          (int((x+w)/scale_percent), int( (y+h)/scale_percent ) ), (120, 255, 0), 10)
            print(w*h)
            
            crop_img = image[y:y+h, 0:x+w]     
            # cv2.imshow("cropped", crop_img)
            
            crop_img = frame_original[int(y/scale_percent):int((y+h+5)/scale_percent), 0:]           
            # crop_img = image[y:y+h+10, 0:]  
            
            # cv2.imshow("cropped", crop_img) 
            # cv2.imwrite('crop_img.png',crop_img)
            
            # cv2.waitKey(0)
            
            stamps.append(crop_img)
    
    # print ("******************************************************* \n      Stamp count = ",len(stamps))
    
    
    
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
        
        
        crop_img=cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
        rect,crop_img = cv2.threshold(crop_img,0,255,cv2.THRESH_BINARY)
        kernel = np.ones((7,7),np.uint8)
        crop_img = cv2.erode(crop_img, kernel, iterations = 1) 
        
        #configuring parameters for tesseract
        custom_config = r'--oem 3 --psm 6  ' #-ctessedit_char_blacklist= 0123456789
        
        # chekc the text next to stamp
        text = pytesseract.image_to_string(crop_img, lang = 'kor', config=custom_config).strip() 
        text=re.sub('[A-Za-z0-9]+', '', text) 
        
        
        
        # now feeding image to tesseract
        details = pytesseract.image_to_data(crop_img, output_type=pytesseract.Output.DICT, config=custom_config, lang="kor")
        
        # print(details.keys())
        total_boxes = len(details['text'])
        for sequence_number in range(total_boxes):
            
        	if ((int(details['conf'][sequence_number]) >50) and (len(details['text'][sequence_number]) > 0)  ):
                
        		(x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], 
                          details['width'][sequence_number],  details['height'][sequence_number])
                
        		threshold_img = cv2.rectangle(crop_img, (x, y), (x + w, y + h), (120, 255, 0), 10) 
          
            
        
        # cv2.imwrite('crop_img2.png',crop_img)
                
        width = int(crop_img.shape[1] * scale_percent )
        height = int(crop_img.shape[0] * scale_percent )
        dim = (width, height)
        
        # display image
        # Maintain output window until user presses a key
        # Destroying present windows on screen
        # cv2.imshow("captured text", cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # actual text next to the stamp after pytesseract locations identification
        text2 = pytesseract.image_to_string(crop_img, lang = 'kor').strip() 
        text2=re.sub('[A-Za-z0-9]+', '', text2)
        
        if(len(text2)==0):
            text2=''.join([i for i in "".join(details["text"]).strip() if  i.isalpha()]).strip()
            
        
        
        
        # cv2.imshow("image", image) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        
        #==================================================================================
        #==================================================================================
        #=================  Read the header text from original image ========================
        #==================================================================================
        #==================================================================================
        
        
        frame_original=cv2.imread('image/normal/'+file)
        #configuring parameters for tesseract
        custom_config = r'--oem 3 --psm 6  ' #-ctessedit_char_blacklist= 0123456789
        
        # header_image1 = frame_original[0:frame_original.shape[1],0:int(frame_original.shape[0]/4),]
        header_image = frame_original[0:int(frame_original.shape[0]/5.5),:,]
        
        
        header_image=cv2.cvtColor(header_image,cv2.COLOR_BGR2GRAY)
        rect,header_image = cv2.threshold(header_image,0,255,cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        header_image=cv2.morphologyEx(header_image, cv2.MORPH_OPEN, kernel)
        header_image = cv2.dilate(header_image, np.ones((5,5),np.uint8), iterations = 1)
        rect,header_image = cv2.threshold(header_image,0,255,cv2.THRESH_BINARY)
        header_image = cv2.erode(header_image, np.ones((9,9),np.uint8), iterations = 1)
        
        
        # header_image=cv2.dilate(header_image,kernel,iterations = 1)
        
        
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
        header_text=re.sub('[A-Za-z0-9]+', '', header_text)
        
        
        padding=50
            
        header_image = cv2.rectangle(header_image, (np.min(temp.left),np.min(temp.top) ), 
                                     ( np.max(temp.left)+np.max(temp.width)   , np.max(temp.top)+np.max(temp.height)), (120, 255, 0), 10)
        
        image_withstampLocator=cv2.rectangle(image_withstampLocator, (np.min(temp.left),np.min(temp.top) ), 
                                      ( np.max(temp.left)+np.max(temp.width)   , np.max(temp.top)+np.max(temp.height)), (120, 255, 0), 10)
        
        width = int(header_image.shape[1] * 0.2 )
        height = int(header_image.shape[0] * 0.2 )
        dim = (width, height)
               
        # cv2.imshow("header_image", cv2.resize(header_image, dim, interpolation = cv2.INTER_AREA))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        #==================================================================================
        #==================================================================================
        #============  compare the text next to the stamp and header text =================
        #==================================================================================
        #==================================================================================
        
        # # print(list(temp.text).join(""))
        # print(header_text, "\n" ,      text2)
        # from difflib import SequenceMatcher
        # print(SequenceMatcher(None, header_text, text2).ratio())
        
        # print(header_text, "\n" ,      text)
        # from difflib import SequenceMatcher
        # print(SequenceMatcher(None, header_text, text).ratio())
        
        from fuzzywuzzy import fuzz
        print(header_text, "\n" ,text2)
        print(fuzz.token_set_ratio(header_text, text2))
        
        stringmatches= (fuzz.token_set_ratio(header_text, text2)>0)
    
    #==================================================================================
    #==================================================================================
    
    
    if(len(stamps)>0 and stringmatches):
        print("--------------> normal")
        cv2.imwrite('result/normal/normal/'+file,image_withstampLocator)
    else:
        print("--------------> not normal")        
        cv2.imwrite('result/normal/notnormal/'+file,image_withstampLocator)
        
        
        
        
        
        
        
        
        
        
        