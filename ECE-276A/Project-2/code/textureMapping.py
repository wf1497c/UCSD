import numpy as np
import matplotlib.pyplot as plt
from torch import float64
from pr2_utils import compute_stereo
import transformation
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
        image_l, image_l_gray, disparity_map = compute_stereo(path_l, path_r)

        pixel_uv = []
        xyz_w = []
        RGB = []
        #print(np.sum(disparity_map >= depth_threshold))
        for u in range(disparity_map.shape[0]):
            for v in range(disparity_map.shape[1]):
                if disparity_map[u,v] >= depth_threshold:
                    pixel_value = image_l[u,v,:]
                    z = fsu * baseline / disparity_map[u,v]#baseline * fsu / disparity_map[u,v]
                    y = z * (u - cu) / fsu
                    x = z * (v - cv) / fsv
                    pixel_uv.append([v,u])
                    xyz_w.append([x,y,z])
                    RGB.append(pixel_value)

        return image_l_gray, np.array(pixel_uv), np.array(xyz_w), np.array(RGB, dtype=np.uint8) #optical frame?

