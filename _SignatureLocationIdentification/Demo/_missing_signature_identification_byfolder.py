# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:35:19 2021

@author: Henock
"""

 

import cv2    
import numpy as np
import pandas as pd
import pytesseract   
from scipy.spatial import distance 
from os import walk

_, _, filenames = next(walk("image/notnormal"))

for file in filenames:        
    print(file)    
    
    isnormal=False;
         
    
    # frame = cv2.imread('TestImages/ct2.png') 
    # frame = cv2.imread('tempimage/1 (19).jpg') 
    frame = cv2.imread('image/notnormal/'+file) 
    
    # frame=cv2.imread("sample.jpg")
    frame_original=frame 
    
    #resize image  
    img=frame
    scale_percent = 500/img.shape[0] # percent of original size
    width = int(img.shape[1] * scale_percent )
    height = int(img.shape[0] * scale_percent )
    dim = (width, height)
    frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    
    #============================================================
    #============================================================
    # identify the horizontal line
    #============================================================
    #============================================================
    
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
    
    
    #============================================================
    #============================================================
    # use tesseract to identify the characters and the words
    #============================================================
    #============================================================
    
    
    #configuring parameters for tesseract
    custom_config = r'--oem 3 --psm 6  ' #-ctessedit_char_blacklist= 0123456789
    
    # chekc the text next to stamp
    text = pytesseract.image_to_string(crop_img, lang = 'kor', config=custom_config).strip() 
    boxes =pytesseract.image_to_boxes(crop_img, lang = 'kor', config=custom_config) 
    details = pd.DataFrame(pytesseract.image_to_data(crop_img, output_type=pytesseract.Output.DICT, config=custom_config, lang="kor"))
    
    # h, w, _ = crop_img.shape
    h, w= crop_img.shape
    
    lines=boxes.splitlines() 
    
    
    previousBox=lines[0].split(' ')
    nextBox=lines[1].split(' ')
        
    
    padding=10
    temptext=""
    
    
    TextList=[]
    LocationList=[]
    _indexOfBC=-1
    
    for i in range(len(lines)-1):   
        
        
        currentBox=lines[i].split(' ')
        nextBox=lines[i+1].split(' ')
        
        # temptext+=currentBox[0].strip()
        # temptext+=''.join([i for i in currentBox[0].strip() if i.isalpha()])
        temptext+=''.join([i for i in currentBox[0].strip() if not i.isdigit()])
        
         #clean up
        temptext=(temptext.replace("대결","").replace("전결","").replace("/","").replace("_","")
                  .replace(".","").replace("결","").replace("전","").replace("-","").replace("”","")
                  .replace("저거" ,"").replace("골","").replace("ㆍ","").replace("고스","")
                  .replace("저그","").replace("보","").replace("장","")
                  ) #.replace("팀" ,"").replace("처","").replace("과","")
         
        diff=(int(nextBox[1]) - int(currentBox[3])) 
         
        if( ( abs(diff) > 50   ) and( (abs(diff) < 0.9*w)) and 
           (nextBox[1]>currentBox[3] or nextBox[3] < currentBox[1]) # to check if the two boxes are not overlaping
           ):    
             
            # locate the benchmark text
            if(('협조자' in temptext.strip())  or ('협소자' in temptext.strip())
               or ('현조자' in temptext.strip()) or ('첩조자' in temptext.strip()  ) 
               or  ('혐조자' in temptext.strip()) ):
                _indexOfBC=len(TextList)
                temptext=(temptext.replace('협조자',"")
                          .replace('협소자',"").replace('현조자',"").strip() 
                          .replace('첩조자',"").strip().replace('혐조자',"").strip() 
                          )
                print(i,"-------****-------------------***")
                
                
            
            LocationList.append( (int(previousBox[1]) , h-int(previousBox[2]) , int(currentBox[3]), h-int(currentBox[4])  ) )
            TextList.append(temptext)
            
                
            previousBox=nextBox
            temptext=""
            
     
    LocationList=np.array(LocationList)
    img = cv2.rectangle(crop_img, 
                            # (LocationList[_indexOfBC][0] -padding, LocationList[_indexOfBC][1]+ padding), 
                            (LocationList[_indexOfBC][0]  ,     LocationList[_indexOfBC][1] +50 ),
                            (LocationList[_indexOfBC][0]  + w,  LocationList[_indexOfBC][1] +50 ),
                            (0, 255, 0), 5)
    
    
    boxandtext= pd.DataFrame(LocationList,columns=["x1","y1","x2","y2"])
    boxandtext["text"]=TextList
        
    # "remove lines below the benchmark "
    val=LocationList[_indexOfBC][1] + 0
    boxandtext2=boxandtext[boxandtext.y1 <val ]
    
    
    boxandtext2=boxandtext2.query("text != ''") 
    # draw the boundig boxes
    
    for i in range(boxandtext2.shape[0]):
        img = cv2.rectangle(crop_img, 
                        (boxandtext2.iloc[i,0],boxandtext2.iloc[i,1]), 
                        (boxandtext2.iloc[i,2],boxandtext2.iloc[i,3]),
                        (120, 255, 0), 5)
                     
    
             
            
    
    width = int(w * 3*scale_percent )
    height = int(h * 3*scale_percent ) 
    dim = (width, height)
    # resize image
    crop_img2 = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) 
    
    # show annotated image and wait for keypress
    # cv2.imshow("crop_img2", crop_img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    
    
    # cv2.imwrite('cropped/crop_img'+file,crop_img)
    cv2.imwrite('testcrop_img.png',crop_img)
    
    
    
    #============================================================
    #============================================================
     
    
    indexes=list(np.arange(boxandtext2.shape[0]))
    boxandtext2=boxandtext2.sort_values(by=["x1","y1"]).reset_index(drop=True)
    
    if(boxandtext2.shape[0]%2 !=0):
        print(" ---------------------- Not Normal ---------------------- ")
    else:
        
        for j in range(boxandtext2.shape[0]):
            
            print(j,"--------- j ",indexes)
                
            if (j in indexes):        
                
                distances=[] 
                temp_dist_index=[]
                
                # for i in range(boxandtext2.shape[0]):
                for i in indexes:
                    a=(boxandtext2.x2[j], boxandtext2.y1[j])
                    b=(boxandtext2.x1[i], boxandtext2.y1[i])            
                    
                    
                    if(i==j):
                        distances.append(pow(10, 7))
                    else :
                        distances.append(distance.euclidean(a, b))
                        
                    temp_dist_index.append(i)
                    # print(i,j," ",boxandtext2.text[i],distance.euclidean(a, b),distances,i==j)
                
                
                closest=temp_dist_index[np.argmin(distances)]        
                # print("closest",closest)  
                
                # limit distance check        
                horizontaldist= abs(boxandtext2.x1[closest] - boxandtext2.x2[j])
                verticaldist= abs (boxandtext2.y2[closest] - boxandtext2.y1[j])
                # print("horizontaldist  ==> ",abs(horizontaldist),horizontaldist<450)
                
                if(horizontaldist<450 and verticaldist < 130 ):                    
                    
                    crop_img2=cv2.arrowedLine(crop_img,
                              (boxandtext2.x2[j], boxandtext2.y1[j]),
                              (boxandtext2.x1[closest], boxandtext2.y1[closest]),
                             
                              (120, 255, 120), 10)
                    
                    # resize image
                    crop_img2 = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) 
                    
                    # # show annotated image and wait for keypress
                    # cv2.imshow("crop_img2", crop_img2)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows() 
                     
        
        
                    indexes.remove(j)
                    indexes.remove(closest)
                     
                    print(indexes, " removed ", j, closest)
    
    if(len(indexes) !=0):
        print(" ---------------------- Not Normal ---------------------- ")
    else:
        print(" ********************** Normal ********************** ")
        isnormal=True
        
    # resize image
    crop_img2 = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) 
    
    # # show annotated image and wait for keypress
    # cv2.imshow("crop_img2", crop_img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    
    if(isnormal):
        cv2.imwrite('cropped/notnormal/normal/'+file,crop_img)
    else:
        cv2.imwrite('cropped/notnormal/notnormal/'+file,crop_img)
        
    

