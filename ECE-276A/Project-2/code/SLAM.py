from json import encoder
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from pr2_utils import read_data_from_csv
import math
import transformation
import particle
import occupancy_grid_map

class getData():
    def __init__(self):
        lidar_path = os.path.abspath('./Lidar/lidar.csv')
        fog_path = os.path.abspath('./FOG/fog.csv')
        encoder_path = os.path.abspath('./Encoder/encoder.csv')

        self.lidar_timestamp, self.lidar_data = read_data_from_csv(lidar_path)
        self.fog_timestamp, self.fog_data = read_data_from_csv(fog_path)
        self.encoder_timestamp, self.encoder_data = read_data_from_csv(encoder_path)

        self.fog_data, self.fog_timestamp = self.synchronize(self.fog_data, self.encoder_data, self.fog_timestamp)
        self.v = self.linear_velocity() 
        self.w = self.angular_velocity()
        self.yaw = self.yaw_t()
        self.delta_pose = self.delta_pose_t()

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
            
            syn_fog_list.insert(0,fog_data[0,j])                                            # sum of delta RPY at first time step should be zero
            syn_fog.append(syn_fog_list[0:len(syn_fog_list)-1])
            syn_fog_timestamp.append(syn_fog_t_list)

        return np.array(syn_fog).T, np.array(syn_fog_timestamp).T


    def linear_velocity(self):
        '''
        average velocity from l/r encoders
        '''
        encoder_data = self.encoder_data
        encoder_timestamp = self.encoder_timestamp
        time_diff = [pow(10,-9)*(encoder_timestamp[i] - encoder_timestamp[i-1]) 
                    for i in range(1,len(encoder_timestamp))]
        encoder_diff = np.array([(encoder_data[i] - encoder_data[i-1]) for i in range(1,len(encoder_data))])
        ld = 0.623479
        rd = 0.622806
        vl = math.pi * ld * encoder_diff[:,0] / 4096 / time_diff
        vr = math.pi * rd * encoder_diff[:,1] / 4096 / time_diff
        v = (vl + vr) / 2
        v = np.insert(v,0,v[0])

        return v

    def angular_velocity(self):
        '''
        compute yaw rate at each timestep
        '''
        delta_yaw = self.fog_data[:,2]
        fog_timestamp = self.fog_timestamp[:,2]
        time_diff = [pow(10,-9)*(fog_timestamp[i] - fog_timestamp[i-1]) 
                    for i in range(1,len(fog_timestamp))]
        w = []
        for i in range(len(time_diff)):
            w.append(delta_yaw[i+1] / time_diff[i])
        w.insert(0,w[0])

        return np.array(w)

    def yaw_t(self):
        delta_yaw = self.fog_data[:,2]
        yaw = np.ones(len(delta_yaw))
        for i in range(1,len(delta_yaw)):
            yaw[i] = yaw[i-1] + delta_yaw[i-1]
        
        return yaw

    def lidar_localization_predict(self, x_t, u_t, w_t, t_interval):
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

    def delta_pose_t(self):
        '''
        Pose change in lidar-based localization with a differential-drive model
        Input:
            x_t: 3d vector represents 2D position and orientation(yaw) of a particle
            v_t: linear velocity - 2d vector
            w_t: angular velocity - 1d vector
            t_interval: time interval
        Output:
            d_pose: 3*N array represents pose change(2D position and orientation(yaw)) of a particle at each timestep 
        '''
        #noise_v = np.random.normal(0,1,1)                                       # gaussian noise
        #noise_w = np.random.normal(0,1,1)
        #w_t += noise_w
        #u_t += noise_v
        timestamp = self.encoder_timestamp
        time_diff = [pow(10,-9)*(timestamp[i] - timestamp[i-1]) 
                    for i in range(1,len(timestamp))]
        time_diff.insert(0,time_diff[0])
        v = self.v
        w = self.w
        yaw = self.yaw
        cos_vec = [math.cos(y) for y in yaw]
        sin_vec = [math.sin(y) for y in yaw]

        d_pose = time_diff * np.vstack([v * cos_vec, v * sin_vec, w])

        return d_pose


    def initializeSLAM(self, num_particles):
        MAP = occupancy_grid_map.initializeMap(0.1,-80,-80,80,80)
        particles = particle.initializeParticles(num_particles)
        TRAJECTORY_w = {}
        TRAJECTORY_w['particle'] = []
        TRAJECTORY_w['odometry'] = []
        
        TRAJECTORY_m = {}
        TRAJECTORY_m['particle'] = []
        TRAJECTORY_m['odometry'] = []
        return MAP, particles, TRAJECTORY_w, TRAJECTORY_m

if __name__ == '__main__':
    data = getData()
    MAP, particles, TRAJECTORY_w, TRAJECTORY_m = data.initializeSLAM(5)

    for i in range(1):#len(data.lidar_data)):

        lidar_angle = np.linspace(-5, 185, 286) / 180 * np.pi
        lidar_range = data.lidar_data[i, :]

        # Remove scan points that are too close or too far
        indValid = [range < 80 and range > 0.1 for range in lidar_range]
        lidar_range = lidar_range[indValid]
        lidar_angle = lidar_angle[indValid]

        # Record trajectories
        delta_pose = data.delta_pose

        if (i == 0):
            pose = particles['poses'][np.argmax(particles['weights']), :]
            
            TRAJECTORY_w['particle'].append(np.expand_dims(pose, axis=0))
            x_m, y_m = transformation.worldToMap(MAP, TRAJECTORY_w['particle'][i][0][0], TRAJECTORY_w['particle'][i][0][1])
            TRAJECTORY_m['particle'].append(np.array([[x_m[0], y_m[0], TRAJECTORY_w['particle'][i][0][2]]]))

            TRAJECTORY_w['odometry'].append(lidar_data[i]['delta_pose'])
            o_x_m, o_y_m = transformation.worldToMap(MAP, TRAJECTORY_w['odometry'][i][0][0], TRAJECTORY_w['odometry'][i][0][1])
            TRAJECTORY_m['odometry'].append(np.array([[o_x_m[0], o_y_m[0], TRAJECTORY_w['odometry'][i][0][2]]]))
        else:
            TRAJECTORY_w['odometry'].append(lidar_data[i]['delta_pose'] + TRAJECTORY_w['odometry'][i - 1])
            o_x_m, o_y_m = transformation.worldToMap(MAP, TRAJECTORY_w['odometry'][i][0][0], TRAJECTORY_w['odometry'][i][0][1])
            TRAJECTORY_m['odometry'].append(np.array([[o_x_m[0], o_y_m[0], TRAJECTORY_w['odometry'][i][0][2]]]))

        print('b')
