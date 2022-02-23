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
    w_T_v = np.hstack((w_T_v, np.array(([x], [y], [0], [1]))))
    
    return w_T_v

def lidarToVehicleTransform(): # correct
    '''
    Output:
        v_T_l: Transformation from lidar to vehicle frame
    '''
    l_R_v = np.array([[0.00130201, 0.796097, 0.605167],
                    [0.999999, -0.000419027, -0.00160026],
                    [-0.00102038, 0.605169, -0.796097]
                    ])
    l_p_v = np.array([[0.8349, -0.0126869, 1.76416, 1]])

    v_T_l = np.vstack([l_R_v, np.zeros(3)])
    v_T_l = np.hstack([v_T_l, l_p_v.T])
    #v_T_l = np.vstack((l_R_v.T, np.zeros((1,3))))
    #v_T_l = np.hstack((v_T_l, np.array([np.append(-l_R_v.T.dot(l_p_v),1),]).T))

    origin = v_T_l.dot(np.array([0,0,0,1]))

    return v_T_l

def polar2cart(lidar_data, lidar_angle):
    '''
    transform polar coordinate system to cartesian system 
    '''
    x = lidar_data * np.cos(lidar_angle)
    y = lidar_data * np.sin(lidar_angle)

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    return x.T, y.T

def lidarToWorldTransform(x_cur, y_cur, yaw_cur):
    '''
    Input:
        x_l, y_l: coordinates in lidar frame
        x_cur, y_cur: vehicle's coordinate in world frame
        yaw_cur: vehicle's yaw angle
    Output:
        w_T_l: transformaion from lidar frame to world frame
    '''
    v_T_l = lidarToVehicleTransform()
    w_T_v = vehicleToWorldTransform(x_cur, y_cur, yaw_cur)
    w_T_l = w_T_v.dot(v_T_l)
    origin = w_T_l.dot(np.array([0,0,0,1]))

    return w_T_l

def lidarToWorld(x_l, y_l, x_cur, y_cur, yaw_cur): ###
    '''
    Input:
        x_l, y_l: coordinates in lidar frame
        x_cur, y_cur: vehicle's coordinate in world frame
        yaw_cur: vehicle's yaw angle
    Output: 
        x_w, y_w: point (x_l,y_l) in world frame
    '''
    w_T_l = lidarToWorldTransform(x_cur,y_cur,yaw_cur)
    coordinates_l = np.vstack((np.vstack((x_l, y_l)), np.zeros((1, x_l.shape[1])), np.ones((1, x_l.shape[1]))))
    coordinates_w = w_T_l.dot(coordinates_l)
    origin = w_T_l.dot(np.array([0,0,0,1]))

    x_w = coordinates_w[0, :]
    y_w = coordinates_w[1, :]
    z_w = coordinates_w[2, :]
    
    x_w = x_w[:, np.newaxis]
    y_w = y_w[:, np.newaxis]
    z_w = z_w[:, np.newaxis]
    
    # remove scans that are too close to the ground
    indValid = (z_w > 0.1)
    x_w = x_w[indValid]
    y_w = y_w[indValid]
    z_w = z_w[indValid]

    return (x_w, y_w, z_w)

def worldToMap(MAP, x_w, y_w): 
    '''
    transform fromt world frame to occupancy grid map
    Input:
        MAP: occupancy grid map
        x_w, y_w: coordinates in world frame
    '''
    # convert from meters to cells
    x_m = np.ceil((x_w - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    y_m = np.ceil((y_w - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    #indGood = np.logical_and(np.logical_and(np.logical_and((x_m > 1), (y_m > 1)), (x_m < MAP['sizex'])),(y_m < MAP['sizey']))
    
    #x_m = x_m[indGood]
    #y_m = y_m[indGood]
    
    return x_m.astype(np.int), y_m.astype(np.int)
