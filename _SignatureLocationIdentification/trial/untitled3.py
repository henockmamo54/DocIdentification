# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:31:55 2021

@author: Henock
"""

 


import cv2  
import re
import imutils
import numpy as np
import pytesseract  
from matplotlib import pyplot as plt

 
# frame = cv2.imread('TestImages/ct2.png') 
frame = cv2.imread('../TrainTestData/Train/Normal/1 (1).jpg')  
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
        
        crop_img = frame_original[int(y/scale_percent):, 0:]   
        cv2.imwrite('crop_img.png',crop_img)


res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))


# cv2.imshow("crop_img", crop_img) 
# cv2.waitKey(0)
# cv2.destroyAllWindows() 


#configuring parameters for tesseract
custom_config = r'--oem 3 --psm 6  ' #-ctessedit_char_blacklist= 0123456789

# chekc the text next to stamp
text = pytesseract.image_to_string(crop_img, lang = 'kor', config=custom_config).strip() 
boxes =pytesseract.image_to_boxes(crop_img, lang = 'kor', config=custom_config)


h, w, _ = crop_img.shape
# draw the bounding boxes on the image
# for b in boxes.splitlines():
#     b = b.split(' ')
#     img = cv2.rectangle(crop_img, 
#                         (int(b[1]), h - int(b[2])), 
#                         (int(b[3]), h - int(b[4])),
#                         (120, 255, 0), 10)



lines=boxes.splitlines()
tempp=lines[0].split(' ')
tempn=lines[0].split(' ')


b=lines[0].split(' ')
n=lines[1].split(' ')
    
for i in range(len(lines)-1):   
    
    
    
    n=lines[i+1].split(' ')
    
    # if(int(b[3])-int(tempp[1])<30):
    #     tempp=lines[i].split(' ')
    diff=abs(int(b[3])-int(n[1]))
    print(i,diff,b)
    
    
    if(diff>100):
        print("************* draw")
        img = cv2.rectangle(crop_img, 
                        (int(n[1]), h - int(n[2])), 
                        (int(b[3]), h - int(b[4])),
                        (0, 100, 120), 15)
        tempp=lines[i].split(' ')
        
        b=lines[i].split(' ')
        
    # else:
    #     img = cv2.rectangle(crop_img, 
    #                     (int(tempp[1]), h - int(tempp[2])), 
    #                     (int(b[3]), h - int(b[4])),
    #                     (0, 100, 120), 15)
        
        
    

width = int(w * 3*scale_percent )
height = int(h * 3*scale_percent ) 
dim = (width, height)
# resize image
crop_img2 = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) 

# show annotated image and wait for keypress
cv2.imshow("crop_img2", crop_img2)
cv2.waitKey(0)
cv2.destroyAllWindows() 




# text2=text


# text=re.sub('[A-Za-z0-9]+', '', text) 
# text=text.strip().replace('/','').strip()

# text=text[0:text.index('협조자')]
# text=" ".join(text.split())
# ppp=(text.split(" "))



# pp=np.array(text.split("\n"))                   


# crop_img_orginal= crop_img
# # # get grayscale image
# # crop_img=cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
# # #thresholding
# # crop_img=cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# # # noise removal
# # crop_img=cv2.blur(crop_img,(5,5)) 


        
# # width = int(3*crop_img.shape[1] * scale_percent )
# # height = int(3*crop_img.shape[0] * scale_percent )
# # dim = (width, height)
# # cv2.imshow("crop_img", cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA))
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# # now feeding image to tesseract
# details = pytesseract.image_to_data(crop_img, output_type=pytesseract.Output.DICT, config=custom_config, lang="kor")
# # print(details.keys())
# total_boxes = len(details['text'])
# for sequence_number in range(total_boxes):
#  	if ((int(details['conf'][sequence_number]) >50)  
#       and (len(details['text'][sequence_number]) > 0) 
#       ):
#           (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], 
#                   details['width'][sequence_number],  details['height'][sequence_number])
         
#           threshold_img = cv2.rectangle(crop_img, (x, y), (x + w, y + h), (120, 255, 0), 10)
  
    

# cv2.imwrite('crop_img2.png',crop_img)
        
# width = int(3*crop_img.shape[1] * scale_percent )
# height = int(3*crop_img.shape[0] * scale_percent )
# dim = (width, height)

# # display image
# # Maintain output window until user presses a key
# # Destroying present windows on screen
# cv2.imshow("captured text", cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) )
# # cv2.imshow("crop_img", crop_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import pandas as pd
# details= pd.DataFrame(details)

