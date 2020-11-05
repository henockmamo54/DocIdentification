# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:44:59 2020

@author: Henock
"""


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import glob
from PIL import Image


# from keras.applications.vgg16 import VGG16
# # load the model
# model = VGG16()


path = 'TrainTestData/Test/Normal'
files = [f for f in glob.glob(path + "**/*.jpg", recursive=True)]
f=files[11]
img=Image.open(f)


plt.imshow(img)

# plt.imshow(np.asarray(img)[:,:,0])
# plt.savefig("r.jpg")


img1 = np.array(img)
figure, plots = plt.subplots(ncols=3, nrows=1)
for i, subplot in zip(range(3), plots):
    temp = np.zeros(img1.shape, dtype='uint8')
    temp[:,:,i] = img1[:,:,i]
    subplot.imshow(temp)
    subplot.set_axis_off()
plt.show()


