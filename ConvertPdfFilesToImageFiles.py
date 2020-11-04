# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:57:15 2020

@author: Henock
"""

import glob
from pdf2image import convert_from_path
import numpy as np
import PIL 


path = 'D:\\WorkPlace\\github projects\\DocIdentification\\OriginalPdfFiles_Translated\\'

files = [f for f in glob.glob(path + "**/*.pdf", recursive=True)]

for f in files:
    print(f)
    pages = convert_from_path(f, 500)     
    
    imgs    = pages
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
        
    # for a vertical stacking it is simple: use vstack
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save( str(f).replace('.pdf','.jpg') )

