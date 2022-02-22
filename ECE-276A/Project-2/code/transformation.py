'''
Frame transformation module
'''

import numpy as np
import matplotlib.pyplot as plt
import math

def rot_z(psi):
    R_z = np.array([[math.cos(psi),    -math.sin(psi),    0],
                    [math.sin(psi),     math.cos(psi),    0],
                    [            0,                 0,    1]
                    ])

    return R_z

def vehicleToWorldTransform(x, y, yaw):
    '''
    Input:
        x, y: 2D position from odometry
        yaw: orientation from fog
    Output:
        w_T_v: Transformation from vehicle to world frame
    '''
    b = 0.623479 / 2            # z-axis translation = left wheel radius
    R = rot_z(yaw)

    w_T_v = np.vstack((R, np.zeros((1,3))))
    w_T_v = np.hstack((w_T_v, np.array(([x], [y], [b], [1]))))

    return w_T_v

def lidarToVehicleTransform():
    '''
    Output:
        v_T_l: Transformation from lidar to vehicle frame
    '''
    l_R_v = np.array([[0.00130201, 0.796097, 0.605167],
                    [0.999999, -0.000419027, -0.00160026],
                    [-0.00102038, 0.605169, -0.796097]
                    ])
    l_p_v = np.array([0.8349, -0.0126869, 1.76416])

    v_T_l = np.vstack((l_R_v.T, np.zeros((1,3))))
    v_T_l = np.hstack((v_T_l, np.array([np.append(-l_R_v.T.dot(l_p_v),1),]).T))

    return v_T_l

lidarToVehicleTransform()
print('a')