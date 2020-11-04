# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:47:07 2020

@author: Henock
"""


import glob
from pdf2image import convert_from_path
import numpy as np
import PIL
from PIL import Image



path = 'D:\\WorkPlace\\github projects\\DocIdentification\\OriginalPdfFiles_Translated\\'

files = [f for f in glob.glob(path + "**/*.pdf", recursive=True)]

for f in files:
    print(f)
    pages = convert_from_path(f, 500)
    
    
    
    # for index, page in enumerate(pages):
    #     page.save(str(f).replace('.pdf','.jpg'), 'JPEG')
        
    
    
    imgs    = pages
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
        
    # for a vertical stacking it is simple: use vstack
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save( str(f).replace('.pdf','.jpg') )


    
    


 

# from pdf2image import convert_from_path
# pages = convert_from_path('D:\\WorkPlace\\github projects\\DocIdentification\\OriginalPdfFiles_Translated\\normal\\1 (68).pdf', 500)


# for index, page in enumerate(pages):
#     page.save(str(index)+'out.jpg', 'JPEG')
    
    


# import numpy as np
# import PIL
# from PIL import Image

# imgs    = pages
# # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
# min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
# imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# # save that beautiful picture
# imgs_comb = PIL.Image.fromarray( imgs_comb)
# imgs_comb.save( 'Trifecta.jpg' )    

# # for a vertical stacking it is simple: use vstack
# imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
# imgs_comb = PIL.Image.fromarray( imgs_comb)
# imgs_comb.save( 'Trifecta_vertical.jpg' )



# import glob, os
# os.chdir('')
# for file in glob.glob("*.pdf"):
#     print(file)
    

import glob

path = 'D:\\WorkPlace\\github projects\\DocIdentification\\OriginalPdfFiles_Translated\\'

files = [f for f in glob.glob(path + "**/*.pdf", recursive=True)]

for f in files:
    print(f)
    