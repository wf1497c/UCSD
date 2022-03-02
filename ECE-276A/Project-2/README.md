Project 2: Particle Filter SLAM

Main modules

1. SLAM: main part of codes. Implementation of particle filter SLAM and texture mapping
2. pr2_utils: compute_stereo function for texture mapping and bresenham2D function for free grid determination were used
3. transformation: includes all transformation function needed in this project, such as lidar to world frame, world frame to ouccupancy grid, optical frame to vehicle frame, and etc.
4. particle: includes prediction, update, resampling, and initialization functions of particle filter implementation
5. occupancy_grid_map: includes initialization and update map functions
6. textureMapping: includes image to world frame transformation and focal length computation

Execution

Only need to execute SLAM.py 