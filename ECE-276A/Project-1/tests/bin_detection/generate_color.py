#%%
import os 
import numpy as np
import cv2
from roipoly import RoiPoly, MultiRoi
import glob 
import matplotlib.pyplot as plt

folder = 'data/training/'

X_blue = np.empty([1,3], dtype = np.int32)
X_neg = np.empty([1,3], dtype = np.int32)

files = os.listdir(folder)
for i in range(16,32): 
    file = files[i]
    img = cv2.imread(os.path.join(folder + file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    fig,ax = plt.subplots()
    ax.imshow(img)
# %%
