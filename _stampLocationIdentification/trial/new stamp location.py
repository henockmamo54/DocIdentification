# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 22:08:06 2020

@author: Henock
"""
 

# importing modules
import cv2
import pytesseract
import numpy as np
import IMPM

# pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'

# reading image using opencv
# image = cv2.imread("test_recepit.png")
# image = cv2.imread("korean9.jpg")


# ===========================================================================================
# ===========================================================================================


# ===========================================================================================
# ===========================================================================================

image = cv2.imread('../TrainTestData/Train/Normal/1 (19).jpg')  

#resize image  500,353
img=image
scale_percent = 1000/img.shape[0] # percent of original size
width = int(img.shape[1] * scale_percent )
height = int(img.shape[0] * scale_percent )
# width = 353
# height = 500
dim = (width, height)

# resize image
image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 


gray = IMPM.get_grayscale(image)
thresh = IMPM.thresholding(gray)
opening = IMPM.opening(gray)
canny = IMPM.canny(gray)


# # display image
# cv2.imshow("gray image", gray)
# cv2.imshow("thresh image", thresh)
# cv2.imshow("opening image", opening)
# cv2.imshow("canny image", canny)
# # Maintain output window until user presses a key
# cv2.waitKey(0)
# # Destroying present windows on screen
# cv2.destroyAllWindows()







text = pytesseract.image_to_string(image, lang = 'kor')

# #converting image into gray scale image
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # converting it to binary image by Thresholding
# # this step is require if you have colored image because if you skip this part 
# # then tesseract won't able to detect text correctly and this will give incorrect result
# threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
# # # display image
# # cv2.imshow("threshold image", threshold_img)
# # # Maintain output window until user presses a key
# # cv2.waitKey(0)
# # # Destroying present windows on screen
# # cv2.destroyAllWindows()

threshold_img=image

#configuring parameters for tesseract
custom_config = r'--oem 3 --psm 6 -c tessedit_char_blacklist= 0123456789'
# now feeding image to tesseract
details = pytesseract.image_to_data(threshold_img, output_type=pytesseract.Output.DICT, config=custom_config, lang="kor")#
print(details.keys())



import pandas as pd
temp=pd.DataFrame(details) 
temp["test_len"] = 0
for i in range(temp.shape[0]):
    temp["test_len"][i]=len(temp.text[i])    
temp= temp[temp.test_len>1]  
temp= temp[temp.conf>30]

# details=temp.to_dict()

total_boxes = len(details['text'])
for sequence_number in range(total_boxes):
	if ((int(details['conf'][sequence_number]) >50)  
     and (len(details['text'][sequence_number]) > 0) 
     ):
		(x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], 
                  details['width'][sequence_number],  details['height'][sequence_number])
		threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (120, 255, 0), 2)
# display image
cv2.imshow("captured text", threshold_img)
# Maintain output window until user presses a key
cv2.waitKey(0)
# Destroying present windows on screen
cv2.destroyAllWindows()


