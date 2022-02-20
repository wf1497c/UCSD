# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 18:38:28 2022

@author: wf149
"""

'''
: Code to generate color data from training images.
: We use roipoly to hand-label blue-recycle bin data as 1 and other pixels as 0
'''

import numpy as np 
import os, cv2
from roipoly import RoiPoly, MultiRoi
from glob import glob
from matplotlib import pyplot as plt
import pickle
import matplotlib

class GenerateColorData(): 
    
    def __init__(self): 
        '''
        : Initialise parameters and methods
        '''
        pass

    def label_data(self,folder): 
        '''
        : We convert the image to YUV space and then select ROI. 
        : The blue recycle bin regions are selected first as positive examples. 
        : Next we chose other regions as negative examples.
        '''

        n = len(next(os.walk(folder))[2])
        print(f"Number of images : {n}")
        X_blue = np.empty([1,3], dtype = np.int32)
        X_neg = np.empty([1,3], dtype = np.int32)

        files = os.listdir(folder)
        for i in range(34,48): 
            file = files[i]
            img = cv2.imread(os.path.join(folder + file))
            img = self.contrast_stretching(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            
            fig,ax = plt.subplots()
            ax.imshow(img)
            my_roi_1 = RoiPoly(fig  = fig, ax = ax, color = 'b')
            mask = np.asarray(my_roi_1.get_mask(img))
            X_blue = np.concatenate([X_blue, img[mask == 1]], axis = 0)

            fig,ax = plt.subplots()
            ax.imshow(img)
            my_roi_2 = RoiPoly(fig = fig, ax = ax, color = 'r')
            mask_2 = np.asarray(my_roi_2.get_mask(img))
            X_neg = np.concatenate([X_neg, img[mask_2==1]], axis = 0)
            
            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # ax1.imshow(mask)
            # ax2.imshow(mask_2)

        return X_blue, X_neg
    
    def contrast_stretching(self, img):
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        img = cv2.LUT(img, table)
        
        return img
        
if __name__ == '__main__':
    
    matplotlib.use('Qt5Agg')
    folder = 'data/training/'
    data_generator = GenerateColorData()
    with open('full_yuv_data.pkl', 'ab') as f: 
        X_pos, X_neg = data_generator.label_data(folder)
        pickle.dump([X_pos, X_neg], f)

    print(f"Number of samples in blue recycle bin : {X_pos.shape}")
    print(f"Number of samples in non blue recycle bin: {X_neg.shape}")