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
import transformation


class stereoModel():
    def __init__(self):
        self.focal_length = self.get_focal_length()

    def get_focal_length(self):
        K = np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02],
                    [0.,7.7537235550066748e+02, 2.5718049049377441e+02],
                    [0., 0., 1.]
                    ])# from projection_matrix
        Ks = np.array([[8.1690378992770002e+02, 5.0510166700000003e-01,6.0850726281690004e+02],
                    [0., 8.1156803828490001e+02,2.6347599764440002e+02],
                    [0., 0., 1.]
                    ])# camera_matrix
        Ff = np.array([[-1,0,0],
                    [0,-1,0],
                    [0,0,1]
                    ])
        Kf = np.linalg.inv(Ks.dot(Ff)).dot(K)
        f = -Kf[0,0]

        return f * 1000
    
    def imageToWorld(self, path_l, path_r, depth_threshold):
        baseline = 0.475143600050775
        cu = 6.1947309112548828e+02
        fsu = 7.7537235550066748e+02
        cv = 2.5718049049377441e+02
        fsv = 7.7537235550066748e+02
        focal_length = self.focal_length
        image_l_gray, image_r_gray, disparity_map = compute_stereo(path_l, path_r)

        pixel_uv = []
        xyz_w = []
        print(np.sum(disparity_map >= depth_threshold))
        for u in range(disparity_map.shape[0]):
            for v in range(disparity_map.shape[1]):
                if disparity_map[u,v] >= depth_threshold:
                    z = fsu * baseline / disparity_map[u,v]#baseline * focal_length / disparity_map[u,v]
                    y = z * (u - cu) / fsu
                    x = z * (v - cv) / fsv
                    pixel_uv.append([v,u])
                    xyz_w.append([x,y,z])

        return image_l_gray, np.array(pixel_uv), np.array(xyz_w) #optical frame?

if __name__ == '__main__':
    path_l = 'stereo_left/1544582648735466220.png'
    path_r = 'stereo_right/1544582648735466220.png'
    s = stereoModel()
    image_l_gray, pixel_uv, pixel_xyz_w = s.imageToWorld(path_l, path_r, 496)
    #pixel_xyz_w = pixel_xyz_w[pixel_xyz_w[:,:,2] >= 50]
    #dist = np.sqrt(pixel_xyz_w[:,1]**2 + pixel_xyz_w[:,0]**2)
    dist = [pixel_xyz_w[:,2]]
    depth_threshold = 50
    #print(np.sum(dist > depth_threshold))
    MAP = occupancy_grid_map.initializeMap(0.1,-20,-20,20,20)
    (x_w, y_w, z_w) = transformation.opticalToWorld(pixel_xyz_w[:,0], pixel_xyz_w[:,1], 0.06,-0.01,0.03)
    x_m, y_m = transformation.worldToMap(MAP, x_w, y_w)

    #plt.plot(pixel_xyz_w[:,2])
    #plt.savefig('height')

    
    #plt.imshow(image_l_gray)
    #print(len(pixel_uv[:,0]), len(pixel_uv[:,1]))
    plt.imshow(MAP['plot'])
    plt.scatter(x_m, y_m, c='r', s=2)
    plt.savefig('Disparity on map')

