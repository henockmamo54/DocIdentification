# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:12:55 2021

@author: Henock
"""
 
import cv2 
import numpy as np 
import pytesseract 
import pandas as pd
import re
  
image = cv2.imread('image/normal/1 (76).jpg') 
frame_original=image  
 
# resize image
scale_percent = 900/image.shape[0] # percent of original size
width = int(image.shape[1] * scale_percent )
height = int(image.shape[0] * scale_percent )
dim = (width, height)
image= cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
 


# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # define range color in HSV
thresh1 = cv2.inRange (hsv, np.array([0, 0, 180]), np.array([0, 0, 204])) 
thresh2 = cv2.inRange (hsv, np.array([0,0,160]), np.array([180,8,216])) 
thresh= thresh1 |thresh2 

rect,thresh = cv2.threshold(thresh,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresh = cv2.erode(thresh, np.ones((3,3),np.uint8), iterations = 1)

## show annotated image and wait for keypress
cv2.imshow("crop_img2", cv2.resize(thresh, (width,height), interpolation = cv2.INTER_AREA) )
cv2.waitKey(0)
cv2.destroyAllWindows() 

#==================================================================================
#==================================================================================
#====================  Find the location of the horizontal bar ====================
#==================================================================================
#==================================================================================

# find contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
mask = np.ones(image.shape[:2], dtype="uint8") * 255

horizontalbars=[]

for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # print(w*h)
    if w*h>1000:
    # if w>width*0.6 and h<10 and h>1 :
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -2)
        print(w*h,y)
        
        crop_img = image[y:y+h, 0:x+w]     
        # cv2.imshow("cropped", crop_img)
        
        crop_img = frame_original[int(y/scale_percent):int((y+h+5)/scale_percent), 0:]           
        # crop_img = image[y:y+h+10, 0:]  
        
        # cv2.imshow("cropped", crop_img) 
        cv2.imwrite('crop_img.png',crop_img)
        
        # cv2.waitKey(0)
        
        horizontalbars.append([crop_img, int((y+h)/scale_percent)])

if(len(horizontalbars)!=1):
    print("Horiztontal bar not found")
else:
    print("horizontal bar found")
    imagebelowthebar= cv2.cvtColor(frame_original,cv2.COLOR_BGR2GRAY)[horizontalbars[0][1]:,] 
    
    # rect,imagebelowthebar = cv2.threshold(imagebelowthebar,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    rect,imagebelowthebar = cv2.threshold(imagebelowthebar,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8) 
    # imagebelowthebar = cv2.dilate(imagebelowthebar, np.ones((2,2),np.uint8), iterations = 1)
    imagebelowthebar = cv2.erode(imagebelowthebar, np.ones((5,5),np.uint8), iterations = 1)
    # imagebelowthebar = cv2.morphologyEx(imagebelowthebar, cv2.MORPH_OPEN, kernel) 


    
    
    # show annotated Eimage and wait for keypress
    # cv2.imshow("crop_img2", cv2.resize(imagebelowthebar, (int(imagebelowthebar.shape[1]*2*scale_percent),
    #                                             int(imagebelowthebar.shape[0]*2*scale_percent)), interpolation = cv2.INTER_AREA) )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    
    # #configuring parameters for tesseract
    custom_config = r'--oem 3 --psm 6  ' #-ctessedit_char_blacklist= 0123456789
    
    # chekc the text next to stamp
    text = pytesseract.image_to_string(imagebelowthebar, lang = 'kor', config=custom_config).strip() 
    boxes =pytesseract.image_to_boxes(imagebelowthebar, lang = 'kor', config=custom_config) 
    details = pd.DataFrame(pytesseract.image_to_data(imagebelowthebar, output_type=pytesseract.Output.DICT, config=custom_config, lang="kor"))
    temp= pd.DataFrame( details)
    
    boxes_array=[]    
    lines=boxes.splitlines() 
    
    for i in (lines):
        boxes_array.append(i.split(' '))
    boxes_array=pd.DataFrame(np.array(boxes_array),columns=["text","x1","y1","x2","y2","c"])
    boxes_array.y1= imagebelowthebar.shape[0]- boxes_array.y1.astype(int)
    boxes_array.y2= imagebelowthebar.shape[0]- boxes_array.y2.astype(int)
    boxes_array.x1=boxes_array.x1.astype(int)
    boxes_array.x2=boxes_array.x2.astype(int)
    
    boxes_array["Diff"]=0
    
    
    temp_text=""
    previous_index=0
    padding=15
    
    
    
    for i in range( boxes_array.shape[0]-1):
        
        # imagebelowthebar=cv2.rectangle(imagebelowthebar, (boxes_array.x1[i], boxes_array.y1[i]), 
        #                                (boxes_array.x2[i],boxes_array.y2[i]), (0, 0, 255), 5)
        
        
        
        diff= abs(boxes_array.x1[i+1]- boxes_array.x2[i])
        boxes_array["Diff"][i]=diff
        
        # temp_text += boxes_array.text[i]
        temp_text +=''.join([i for i in boxes_array.text[i].strip() if not i.isdigit()])
        #clean up
        temp_text=(temp_text.replace("/","").replace("_","").replace(".","")
                   .replace("-","").replace("”","").replace("ㆍ","")
                   .replace("|","").replace(",","")  
                  )  
        
        temp_text=(temp_text.replace("전결","").replace("대결","")                   
                  )  
    
        
        if(~( (diff < 0.8*image.shape[1]) and diff<40)):
            if(len(temp_text)>0):                    
                print("test ",temp_text)
                imagebelowthebar=cv2.rectangle(imagebelowthebar, (boxes_array.x1[previous_index]-padding, padding+boxes_array.y1[previous_index]), 
                                           (padding+boxes_array.x2[i],boxes_array.y2[i] - padding), (0, 0, 255), 5)
            previous_index=i+1
            temp_text=""
            
        
            
        

    cv2.imshow("crop_img2", cv2.resize(imagebelowthebar, (int(imagebelowthebar.shape[1]*1.5*scale_percent),
                                                int(imagebelowthebar.shape[0]*1.5*scale_percent)), interpolation = cv2.INTER_AREA) )
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

cv2.waitKey(0)
cv2.destroyAllWindows() 



