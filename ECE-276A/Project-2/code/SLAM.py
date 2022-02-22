import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from pr2_utils import read_data_from_csv
import math
import transformation

class data_process():
    def __init__(self):
        lidar_path = os.path.abspath('./Lidar/lidar.csv')
        fog_path = os.path.abspath('./FOG/fog.csv')
        encoder_path = os.path.abspath('./Encoder/encoder.csv')

        self.lidar_timestamp, self.lidar_data = read_data_from_csv(lidar_path)
        self.fog_timestamp, self.fog_data = read_data_from_csv(fog_path)
        self.encoder_timestamp, self.encoder_data = read_data_from_csv(encoder_path)

        self.synchronize(self.fog_data, self.encoder_data, self.fog_timestamp) 

        pass

    def synchronize(self, fog_data, encoder_data, fog_timestamp):
        '''
        synchronize fog and encoder data to make vector lengths equivalent
        '''
        syn_fog = []
        syn_fog_timestamp = []
        for j in range(3):
            syn_fog_list = []
            syn_fog_t_list = []
            for i in range(len(encoder_data)):
                syn_fog_list.append(sum(fog_data[i*10:(i+1)*10,j]))             # sum within every 10 samples 
                syn_fog_t_list.append(fog_timestamp[i*10])
            
            syn_fog_list.insert(0,0)                                            # sum of delta RPY at first time step should be zero
            syn_fog.append(syn_fog_list[0:len(syn_fog_list)-1])
            syn_fog_timestamp.append(syn_fog_t_list)
        
        self.fog_data = np.array(syn_fog).T
        self.fog_timestamp = np.array(syn_fog_timestamp).T

        pass

    def linear_velocity(self, encoder_timestamp, encoder_data):
        '''
        average velocity from l/r encoders
        '''
        time_diff = [pow(10,-8)*(encoder_timestamp[i] - encoder_timestamp[i-1]) 
                    for i in range(1,len(encoder_timestamp))]
        time_diff.insert(0,time_diff[0])                                        # append time diff to remain the same length
        ld = 0.623479
        rd = 0.622806
        vl = math.pi * ld * encoder_data[:,0] / 360 / time_diff
        vr = math.pi * rd * encoder_data[:,1] / 360 / time_diff
        v = (vl + vr) / 2

        return v
    
    def lidar_based_localization_prediction(self, x_t, u_t, w_t, t_interval):
        '''
        Prediction step in lidar-based localization with a differential-drive model
        Input:
            x_t: 3d vector represents 2D position and orientation(yaw) of a particle
            u_t: linear velocity - 2d vector
            w_t: angular velocity - 1d vector
            t_interval: time interval
        Output:
            mu_t_1: 3d vector represents 2D position and orientation(yaw) of a particle in next time step
        '''
        noise_v = np.random.normal(0,1,1)                                       # gaussian noise
        noise_w = np.random.normal(0,1,1)
        w_t += noise_w
        u_t += noise_v
        x_t_1 = x_t + t_interval * np.vstack(
            [np.array([[u_t * math.cos(x_t[2]),
            u_t * math.sin(x_t[2])],]).T, 
            w_t])

        return x_t_1

    def transformation_lidar_to_body(self, lidar_data, R_B_to_S, p_B_to_S):
        '''
        Input: FOG/lidar sensor data, rotation matrix, and translation
        Output: data in body frame 
        Rotation and translation was from vehicle to sensor, which are given in Vehicle2Lidar.txt
        '''
        angles = np.linspace(-5, 185, 286) / 180 * np.pi
        ranges = lidar_data[0, :]



if __name__ == '__main__':
    data = data_process()
    vl = data.linear_velocity(data.encoder_timestamp,data.encoder_data)
    print(data.lidar_data.shape)
    