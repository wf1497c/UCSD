import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from pr2_utils import read_data_from_csv
import math
import transformation

def initializeParticles(num = None, n_thresh = None, noise_cov = None):
    if num == None:
        num = 100
    if n_thresh == None:
        n_thresh = 0.1 * num # set threshold to 20% of original number of particles to resample
    if noise_cov == None:
        noise_cov = np.zeros((3,3)) # for debugging purposes
        noise_cov = 0.5 * np.eye(3) # set noise covariances for multivariate Gaussian. This is 10% of the delta_pose movement (check predictParticles)
        noise_cov = np.array([[.1, 0, 0], [0, .1, 0], [0, 0, 0.005]])
    
    PARTICLES = {}
    PARTICLES['num'] = num
        
    PARTICLES['n_thresh'] = n_thresh    # below this value, resample
    PARTICLES['noise_cov'] = noise_cov  # covariances for Gaussian noise in each dimension
    
    PARTICLES['weights'] = np.ones(PARTICLES['num']) / PARTICLES['num'] 
    PARTICLES['poses'] = np.zeros((PARTICLES['num'], 3))

    return PARTICLES

def predictParticles(PARTICLES, d_x, d_y, d_yaw, x_prev, y_prev, yaw_prev):
    
    noise_cov =  np.matmul(PARTICLES['noise_cov'], np.abs(np.array([[d_x, 0, 0], [0, d_y, 0], [0, 0, d_yaw]]))) 
        
    # create hypothesis (particles) poses
    noise = np.random.multivariate_normal([0, 0, 0], noise_cov, PARTICLES['num'])
    PARTICLES['poses'] = noise + np.array([[x_prev, y_prev, yaw_prev]])
    
    # update poses according to deltas
    PARTICLES['poses'] += np.array([[d_x, d_y, d_yaw]])
    return

def updateParticles(PARTICLES, MAP, x_l, y_l, psi, theta):
    
    n_eff = 1 / np.sum(np.square(PARTICLES['weights']))
    
    if (n_eff < PARTICLES['n_thresh']):
        print("resampling!")
        resampleParticles(PARTICLES)
    
    correlations = np.zeros(PARTICLES['num'])
    
    _, plot = cv2.threshold(MAP['plot'], 127, 255, cv2.THRESH_BINARY)
    
    for i in range(PARTICLES['num']):
        x_w, y_w, _ = lidar2world(psi, theta, x_l, y_l, PARTICLES['poses'][i][0], PARTICLES['poses'][i][1], PARTICLES['poses'][i][2])        
        x_m, y_m = world2map(MAP, x_w, y_w)
        
        particle_plot = np.zeros(MAP['plot'].shape)
        particle_plot[y_m, x_m] = [0, 1, 0]

        correlations[i] = np.sum(np.logical_and(plot, particle_plot)) # switched x and y
    
    weights = scipy.special.softmax(correlations - np.max(correlations)) # np.multiply(PARTICLES['weights'], scipy.special.softmax(correlations)) # multiply or add or replace?

    if (np.count_nonzero(correlations) == 0):
        print("ALL ZERO CORRELATIONS")
    
    PARTICLES['weights'] = weights
    
    return

def resampleParticles(PARTICLES):
    # implemented low-variance resampling according to: https://robotics.stackexchange.com/questions/7705/low-variance-resampling-algorithm-for-particle-filter
    
    M = PARTICLES['num']
    new_poses = np.zeros(PARTICLES['poses'].shape)
    
    r = np.random.uniform(0, 1 / M)
    w = PARTICLES['weights'][0]
    i = 0
    j = 0
    
    for m in range(M):
        U = r + m / M
        while (U > w):
            i += 1
            w += PARTICLES['weights'][i]
        new_poses[j, :] = PARTICLES['poses'][i, :]
        j += 1

    PARTICLES['poses'] = new_poses
    PARTICLES['weights'] = np.ones(PARTICLES['num']) / PARTICLES['num'] 

    return