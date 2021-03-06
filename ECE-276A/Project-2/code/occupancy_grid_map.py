import numpy as np
import matplotlib.pyplot as plt
from pr2_utils import bresenham2D
import transformation

def initializeMap(res, xmin, ymin, xmax, ymax, trust = None, optimism = None, occupied_thresh = None, free_thresh = None, confidence_limit = None):
    if trust == None:
        trust = 0.95
    if optimism == None:
        optimism = 1#0.5
    if occupied_thresh == None:
        occupied_thresh = 0.85
    if free_thresh == None:
        free_thresh = 0.5 # prevent over confidence, but not very useful
    if confidence_limit == None:
        confidence_limit = 40
    
    MAP = {}
    MAP['res']   = res      # meters per grid
    MAP['xmin']  = xmin     # meters
    MAP['ymin']  = ymin
    MAP['xmax']  = xmax
    MAP['ymax']  = ymax
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) 
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) 
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float64) 
    
    # Related to log-odds 
    MAP['occupied'] = np.log(trust / (1 - trust))
    MAP['free'] = optimism * np.log((1 - trust) / trust) 
    MAP['confidence_limit'] = confidence_limit

    # Related to occupancy grid
    MAP['occupied_thresh'] = np.log(occupied_thresh / (1 - occupied_thresh))
    MAP['free_thresh'] = np.log(free_thresh / (1 - free_thresh))
    (h, w) = MAP['map'].shape
    MAP['plot'] = np.ones((h, w, 3), np.uint8) * 127
    MAP['texture_map'] = np.zeros((h, w, 3), np.uint8)
    
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
    
    # check whether it's out of range of image size
    indvalid = np.logical_and(np.logical_and(np.logical_and((x_m > 1), (y_m > 1)), (x_m < MAP['sizex'])),(y_m < MAP['sizey']))
    x_m = x_m[indvalid]
    y_m = y_m[indvalid]

    # occupied in map
    MAP['map'][x_m,y_m] += MAP['occupied']#- MAP['free'] # we're going to add the MAP['free'] back in a second
    
    # initialize a mask where we will label the free cells
    x_m = np.append(x_m, x_cur_m) # Must consider robot's current cell
    y_m = np.append(y_m, y_cur_m)
    contours = np.vstack((x_m, y_m)) # SWITCH

    # find the cells that are free by bresenham2D
    
    for i in range(contours.shape[1]):
        hit_x_m, hit_y_m = x_m[i], y_m[i]
        frees = bresenham2D(x_cur_m, y_cur_m, hit_x_m, hit_y_m)

        # prevent out-of-range error. make sure all lidar paths are within map
        indvalid = np.logical_and(np.logical_and(np.logical_and((frees[0,:] > 1), (frees[1,:] > 1)),(frees[0,:] < MAP['sizex'])),(frees[1,:] < MAP['sizey'])) 
        frees = frees[:,indvalid]

        MAP['map'][frees[1,:].astype(int),frees[0,:].astype(int)] += MAP['free']
    
    # update plot
    occupied_grid = MAP['map'][0:x_cur_m+800,0:x_cur_m+800] > 0
    free_grid = MAP['map'][0:x_cur_m+800,0:x_cur_m+800] < 0
    MAP['plot'][0:x_cur_m+800,0:x_cur_m+800][occupied_grid] = [0, 0, 0]
    MAP['plot'][0:x_cur_m+800,0:x_cur_m+800][free_grid] = [255, 255, 255] 
    MAP['plot'][y_m, x_m] = [255, 0, 0] # plot latest lidar scan hits
