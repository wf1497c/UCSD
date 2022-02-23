import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from pr2_utils import read_data_from_csv, bresenham2D
import math
import transformation

def initializeMap(res, xmin, ymin, xmax, ymax, memory = None, trust = None, optimism = None, occupied_thresh = None, free_thresh = None, confidence_limit = None):
    if memory == None:
        memory = 1 # set to value between 0 and 1 if memory is imperfect
    if trust == None:
        trust = 0.8
    if optimism == None:
        optimism = 0.5
    if occupied_thresh == None:
        occupied_thresh = 0.85
    if free_thresh == None:
        free_thresh = 0.2 # 0.5 # 0.25
    if confidence_limit == None:
        confidence_limit = 40 * memory
    
    MAP = {}
    MAP['res']   = res      # meters per grid
    MAP['xmin']  = xmin     # meters
    MAP['ymin']  = ymin
    MAP['xmax']  = xmax
    MAP['ymax']  = ymax
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) # number of horizontal cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) # number of vertical cells
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float64) # DATA TYPE: char or int8
    
    # Related to log-odds 
    MAP['memory'] = memory
    MAP['occupied'] = np.log(trust / (1 - trust))
    MAP['free'] = optimism * np.log((1 - trust) / trust) # Try to be optimistic about exploration, so weight free space
    MAP['confidence_limit'] = confidence_limit

    # Related to occupancy grid
    MAP['occupied_thresh'] = np.log(occupied_thresh / (1 - occupied_thresh))
    MAP['free_thresh'] = np.log(free_thresh / (1 - free_thresh))
    (h, w) = MAP['map'].shape
    MAP['plot'] = np.zeros((h, w, 3), np.uint8) 
    
    return MAP 

def updateMap(MAP, x_w, y_w, x_cur, y_cur):
    '''
    Input:
        MAP: occupany grid map
        x_w, y_w: coordinates of lidar hits in world frame
        x_cur, y_cur: vehicle's current position in world frame
    '''
    # transform lidar hits into map coordinates
    x_m, y_m = transformation.worldToMap(MAP, x_w, y_w)
    
    # transform particles positions into map coordinates
    x_cur_m, y_cur_m = transformation.worldToMap(MAP, x_cur, y_cur)
    #plt.imshow(MAP['plot'])
    #plt.scatter(x_m, y_m, c='r', marker="s")
    #plt.savefig('test.jpg')
    
    # check whether it's out of range of image size
    indvalid = np.logical_and(np.logical_and(np.logical_and((x_m > 1), (y_m > 1)), (x_m < MAP['sizex'])),(y_m < MAP['sizey']))
    x_m = x_m[indvalid]
    y_m = y_m[indvalid]

    MAP['map'] = MAP['map'] * MAP['memory']

    # occupied in map
    MAP['map'][x_m,y_m] += MAP['occupied'] #- MAP['free'] # we're going to add the MAP['free'] back in a second
    
    # initialize a mask where we will label the free cells
    free_grid = np.zeros(MAP['map'].shape).astype(np.int8) 
    x_m = np.append(x_m, x_cur_m) # Must consider robot's current cell
    y_m = np.append(y_m, y_cur_m)
    contours = np.vstack((x_m, y_m)) # SWITCH

    # find the cells that are free, and update the map
    cv2.drawContours(free_grid, [contours.T], -1, MAP['free'], -1) 
    MAP['map'] += free_grid 
    
    # prevent overconfidence
    MAP['map'][MAP['map'] > MAP['confidence_limit']] = MAP['confidence_limit']
    MAP['map'][MAP['map'] < -MAP['confidence_limit']] = -MAP['confidence_limit']
    
    # update plot
    occupied_grid = MAP['map'] > MAP['occupied_thresh']
    free_grid = MAP['map'] < MAP['free_thresh']
    plt.imshow(free_grid)
    plt.savefig("free_grid")
    plt.imshow(occupied_grid)
    plt.savefig("occupied_grid")
    
    MAP['plot'][occupied_grid] = [0, 0, 0]
    MAP['plot'][free_grid] = [255, 255, 255] 
    plt.imshow(MAP['plot'])
    plt.savefig('map_before.png')
    
    MAP['plot'][np.logical_and(np.logical_not(free_grid), np.logical_not(occupied_grid))] = [127, 127, 127]
    
    x_m, y_m = transformation.worldToMap(MAP, x_w, y_w)
    MAP['plot'][y_m, x_m] = [255, 0, 0]    # plot latest lidar scan hits


def test_mapCorrelation():
    _, lidar_data = read_data_from_csv('/home/wf1497c/ML/UCSD/ECE-276A/Project-2/Lidar/lidar.csv')
    angles = np.linspace(-5, 185, 286) / 180 * np.pi
    ranges = lidar_data[0, :]

    # take valid indices
    indValid = np.logical_and((ranges < 80),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # init MAP
    MAP = {}
    MAP['res']   = 0.1 #meters
    MAP['xmin']  = -50  #meters
    MAP['ymin']  = -50
    MAP['xmax']  =  50
    MAP['ymax']  =  50 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
    
    #import pdb
    #pdb.set_trace()
    
    # xy position in the sensor frame
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)
    
    # convert position in the map frame here 
    Y = np.stack((xs0,ys0))
    
    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[indGood],yis[indGood]]=1
        
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

    x_range = np.arange(-0.4,0.4+0.1,0.1)
    y_range = np.arange(-0.4,0.4+0.1,0.1)

    plt.imshow(MAP['map'], cmap = 'gray')
    plt.savefig('map.jpg')
    print('a')

def lidarToWorldFrame(x_l, y_l):
    v_T_l = transformation.lidarToVehicleTransform()
    w_T_v = transformation.vehicleToWorldTransform()
