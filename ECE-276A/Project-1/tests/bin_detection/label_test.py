'''
: Code to generate color data from training images.
: We use roipoly to hand-label blue-recycle bin data as 1 and other pixels as 0
'''

import numpy as np 
import os,cv2
from roipoly import RoiPoly, MultiRoi
from glob import glob
from matplotlib import pyplot as plt
import pickle

class GenerateColorData(): 
    def __init__(self): 
        '''
        : Initialise parameters and methods
        '''
        pass

    def generate_color_data(self,folder): 
        '''
        : We convert the image to RGB space and then select regions of interest. 
        : The blue recycle bin regions are selected first as positive examples. 
        : Next we chose other regions as negative examples.
        '''

        n = len(next(os.walk(folder))[2])
        print(f"Number of images : {n}")
        X_blue = np.empty([1,3], dtype = np.int32)
        X_neg = np.empty([1,3], dtype = np.int32)

        files = os.listdir(folder)
        for i in range(32,50): 
            file = files[i]
            img = cv2.imread(os.path.join(folder + file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig,ax = plt.subplots()
            ax.imshow(img)

            my_roi_1 = RoiPoly(fig  = fig, ax = ax, color = 'b')
            mask = np.asarray(my_roi_1.get_mask(img))
            X_blue = np.concatenate([X_blue, img[mask == 1]], axis = 0)

            fig,ax = plt.subplots()
            ax.imshow(img)
            my_roi_2 = RoiPoly(fig = fig, ax = ax, color = 'r')
            mask = np.asarray(my_roi_2.get_mask(img))
            X_neg = np.concatenate([X_neg, img[mask==1]], axis = 0)

        return X_blue, X_neg

if __name__ == '__main__':
    folder = glob('bin_detection/data/training/')[0]
   
    data_generator = GenerateColorData()
    
    with open('color_data.pkl', 'ab') as f: 
        X_pos, X_neg = data_generator.generate_color_data(folder)
        pickle.dump([X_pos, X_neg], f)

    with open('color_data.pkl', 'rb') as f:
        X_pos = np.empty([1,3], dtype = np.int32)
        X_neg = np.empty([1,3], dtype = np.int32)
        while True: 
            try: 
                X = pickle.load(f)
                X_pos = np.concatenate([X_pos, X[0]], axis = 0)
                X_neg = np.concatenate([X_neg, X[1]], axis = 0)
            except EOFError: 
                break
    print(f"Number of samples in blue recycle bin : {X_pos.shape}")
    print(f"Number of samples in non blue recycle bin: {X_neg.shape}")
    
    with open('full_color_data.pkl', 'wb') as f: 
        pickle.dump([X_pos, X_neg], f)