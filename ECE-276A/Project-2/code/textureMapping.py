from json import encoder
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch import float64
from pr2_utils import compute_stereo
import math
import transformation
import particle
import occupancy_grid_map


class stereoModel():
    def __init__(self):
        self.focal_length = self.get_focal_length()

    def get_focal_length():
        K = np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02],
                    [0.,7.7537235550066748e+02, 2.5718049049377441e+02],
                    [0., 0., 1.]
                    ]) # from projection_matrix
        Ks = np.array([[8.1690378992770002e+02, 5.0510166700000003e-01,6.0850726281690004e+02],
                    [0., 8.1156803828490001e+02,2.6347599764440002e+02],
                    [0., 0., 1.]
                    ]) # camera_matrix
        Ff = np.array([[-1,0,0],
                    [0,-1,0],
                    [0,0,1]
                    ])
        Kf = np.linalg.inv(Ks).dot(np.linalg.inv(Ff)).dot(K)
        f = -Kf[0,0]

        return f

    def 

if __name__ == '__main__':
    path_l = 'code/data/image_left.png'
    path_r = 'code/data/image_left.png'
    compute_stereo(path_l, path_r)
