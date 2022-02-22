import numpy as

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
        confidence_limit = 100 * memory
    
    MAP = {}
    MAP['res']   = res #meters; used to detrmine the number of square cells
    MAP['xmin']  = xmin  #meters
    MAP['ymin']  = ymin
    MAP['xmax']  = xmax
    MAP['ymax']  = ymax
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) # number of horizontal cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) # number of vertical cells
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float64) # contains log odds. DATA TYPE: char or int8
    
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