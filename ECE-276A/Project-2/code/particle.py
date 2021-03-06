import numpy as np
import cv2
from pr2_utils import read_data_from_csv
from scipy import special
import transformation
from time import time

def initializeParticles(num = None, n_thresh = None, noise_cov = None):
    if num == None:
        num = 100
    if n_thresh == None:
        n_thresh = 0.1 * num # set threshold to 10% of original number of particles to resample
    if noise_cov == None:
        noise_cov = np.zeros((3,3))
        noise_cov = 0.5 * np.eye(3) 
        noise_cov = np.array([[.1, 0, 0], [0, .1, 0], [0, 0, 0.005]])
    
    PARTICLES = {}
    PARTICLES['num'] = num
        
    PARTICLES['n_thresh'] = n_thresh    # below this value, resample
    PARTICLES['noise_cov'] = noise_cov  # covariances for Gaussian noise in each dimension
    
    PARTICLES['weights'] = np.ones(PARTICLES['num']) / PARTICLES['num'] 
    PARTICLES['poses'] = np.zeros((PARTICLES['num'], 3))

    return PARTICLES

def predictParticles(PARTICLES, d_x, d_y, d_yaw, x_prev, y_prev, yaw_prev):
    # noise type
    noise_cov =  np.matmul(PARTICLES['noise_cov'], np.abs(np.array([[0.015,0,0],[0,0.015,0],[0,0,0.5]])))#[[d_x, 0, 0], [0, d_y, 0], [0, 0, d_yaw]]))) 
    #noise_cov =  np.matmul(PARTICLES['noise_cov'], np.abs(np.array([[d_x*0.015,0,0],[0,d_y*0.015,0],[0,0,d_yaw*0.5]])))
    #noise_cov =  np.matmul(PARTICLES['noise_cov'], np.abs(np.array([[np.sqrt(abs(d_x))*0.015,0,0],[0,np.sqrt(abs(d_y))*0.015,0],[0,0,np.sqrt(abs(d_yaw))*0.5]])))
    
    # create hypothesis (particles) poses
    noise = np.random.multivariate_normal([0, 0, 0], noise_cov, PARTICLES['num'])
    PARTICLES['poses'] = noise + np.array([[x_prev, y_prev, yaw_prev]])
    
    # update poses according to deltas
    PARTICLES['poses'] += np.array([[d_x, d_y, d_yaw]])
    return

def updateParticles(PARTICLES, MAP, x_l, y_l, TRAJECTORY_m, init=False):
    
    n_eff = 1 / np.sum(np.square(PARTICLES['weights']))
    
    if (n_eff < PARTICLES['n_thresh']):
        print("-----------resampling-----------")
        resampleParticles(PARTICLES)
    
    correlations = np.zeros(PARTICLES['num'])

    if init == False:
        cur_x_m, cur_y_m = int(TRAJECTORY_m['particle'][-1][0]), int(TRAJECTORY_m['particle'][-1][0])
    else:
        cur_x_m, cur_y_m = 200-1, 200-1

    
    plot = MAP['plot']
    if cur_x_m < 800:
        plot = plot[cur_x_m-150:cur_x_m+800,cur_y_m-150:cur_y_m+800,:] ######
    else:
        plot = plot[cur_x_m-800:cur_x_m+800,cur_y_m-800:cur_y_m+800,:]
    
    _, plot = cv2.threshold(plot, 127, 255, cv2.THRESH_BINARY) ####### slow

    map_shape = plot.shape
    particle_plot = np.zeros(map_shape)
    
    for i in range(PARTICLES['num']):
        x_w, y_w, _ = transformation.lidarToWorld(x_l, y_l, PARTICLES['poses'][i][0], PARTICLES['poses'][i][1], PARTICLES['poses'][i][2])        
        x_m, y_m = transformation.worldToMap(MAP, x_w, y_w)

        indvalid = np.logical_and(np.logical_and(np.logical_and((x_m > 1), (y_m > 1)), (x_m < MAP['sizex'])),(y_m < MAP['sizey']))
        if cur_x_m < 800:
            indvalid = np.logical_and(indvalid, np.logical_and(y_m-cur_y_m+150 < 950, x_m-cur_x_m+150 < 950))
            indvalid = np.logical_and(indvalid, np.logical_and(y_m-cur_y_m+150 >=0, x_m-cur_x_m+150 >= 0))
            x_m, y_m = x_m[indvalid], y_m[indvalid]
            particle_plot[y_m-cur_y_m+150, x_m-cur_x_m+150] = [0,1,0]
            correlations[i] = np.sum(np.logical_and(plot, particle_plot)) # too slow
            particle_plot[y_m-cur_y_m+150, x_m-cur_x_m+150] = [0,0,0]
        else:
            indvalid = np.logical_and(indvalid, np.logical_and(y_m-cur_y_m+800 < 1600, x_m-cur_x_m+800 < 1600))
            indvalid = np.logical_and(indvalid, np.logical_and(y_m-cur_y_m+800 >=0, x_m-cur_x_m+800 >= 0))
            x_m, y_m = x_m[indvalid], y_m[indvalid]
            particle_plot[y_m-cur_y_m+150, x_m-cur_x_m+150] = [0,1,0]
            correlations[i] = np.sum(np.logical_and(plot, particle_plot)) # too slow
            particle_plot[y_m-cur_y_m+800, x_m-cur_x_m+800] = [0,0,0]
    
    weights = special.softmax(correlations - np.max(correlations)) 

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