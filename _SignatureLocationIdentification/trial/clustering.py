# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:48:17 2021

@author: Henock
"""


import  cv2 as cv
import numpy as np



img=cv.imread("crop_img3.png")
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


cv.imshow("imread",img)
cv.waitKey(0)
cv.destroyAllWindows()



from sklearn.cluster import DBSCAN


clustering = DBSCAN(eps=3, min_samples=5).fit(img)
clustering.labels_

clustering



