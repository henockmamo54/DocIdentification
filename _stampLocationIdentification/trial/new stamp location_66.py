# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 00:09:46 2020

@author: Henock
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:52:15 2020

@author: Henock
""" 

import numpy as np
import cv2  
import imutils
 

image = cv2.imread('../TrainTestData/Train/Normal/1 (16).jpg')  
frame_original=image 


 
scale_percent = 500/image.shape[0] # percent of original size
width = int(image.shape[1] * scale_percent )
height = int(image.shape[0] * scale_percent )
dim = (width, height)
# resize image
image= cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define range of red color in HSV
lower_red = np.array([0, 20, 100])
upper_red = np.array([346, 255, 255])
thresh_inv = cv2.inRange (hsv, lower_red, upper_red)

# Blur the image
thresh=thresh_inv= cv2.blur(thresh_inv,(3,3)) 

# find contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
mask = np.ones(image.shape[:2], dtype="uint8") * 255

for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    if w*h>1000:
    # if w>width*0.6 and h<10 and h>1 :
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -2)
        print(w*h)
        
        crop_img = image[y:y+h, 0:x+w]     
        cv2.imshow("cropped", crop_img)
        
        crop_img = frame_original[int(y/scale_percent):int((y+h+5)/scale_percent), 0:]           
        # crop_img = image[y:y+h+10, 0:]  
        
        # cv2.imshow("cropped", crop_img) 
        cv2.imwrite('crop_img.png',crop_img)
        
        cv2.waitKey(0)


res_final = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

# # cv2.imshow("gray", gray)
# cv2.imshow("image", image)
# cv2.imshow("boxes", mask)
# cv2.imshow("final image", res_final)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import cv2
import pytesseract
import numpy as np

text = pytesseract.image_to_string(crop_img, lang = 'kor') 

#configuring parameters for tesseract
custom_config = r'--oem 3 --psm 6 -c tessedit_char_blacklist= 0123456789'
# now feeding image to tesseract
details = pytesseract.image_to_data(crop_img, output_type=pytesseract.Output.DICT, config=custom_config, lang="kor")#
# print(details.keys())


total_boxes = len(details['text'])
for sequence_number in range(total_boxes):
	if ((int(details['conf'][sequence_number]) >50)  
     and (len(details['text'][sequence_number]) > 0) 
     ):
		(x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], 
                  details['width'][sequence_number],  details['height'][sequence_number])
		threshold_img = cv2.rectangle(crop_img, (x, y), (x + w, y + h), (120, 255, 0), 10)
  
    

cv2.imwrite('crop_img2.png',crop_img)
        
width = int(crop_img.shape[1] * scale_percent )
height = int(crop_img.shape[0] * scale_percent )
dim = (width, height)

# display image
cv2.imshow("captured text", cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA) )
# Maintain output window until user presses a key
cv2.waitKey(0)
# Destroying present windows on screen
cv2.destroyAllWindows()


text2 = pytesseract.image_to_string(crop_img, lang = 'kor') 






# import pandas as pd
# temp=pd.DataFrame(details) 
# temp["test_len"] = 0
# for i in range(temp.shape[0]):
#     temp["test_len"][i]=len(temp.text[i])    
# temp= temp[temp.test_len>0]  
# temp= temp[temp.conf>30]




