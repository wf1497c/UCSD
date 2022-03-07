import numpy as np
from pr3_utils import *
from scipy.linalg import expm
from src.EKF import EKFSLAM

def initCar(time):
	car = {}
	car['mean'] = np.eye(4) # in SE(3)
	car['covariance'] = 0.01 * np.eye(6)
	car['trajectory'] = np.zeros((4, 4, len(time[0]))) # running history

	return car

def initLandmarks(features):
	num_features = features.shape[1]
	landmarks = {}
	landmarks['mean'] = np.empty(features.shape[:2])
	landmarks['mean'].fill(np.nan)
	landmarks['trajectory'] = np.empty(features.shape)
	landmarks['covariance'] = np.zeros((3, 3, num_features)) 
	
	for i in range(num_features):
		landmarks['covariance'][:, :, i] = 0.01 * np.eye(3)
		
	return landmarks


if __name__ == '__main__':
	# Load the measurements
	filename = "code/data/10.npz"
	time,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

	landmarks = initLandmarks(features)
	car = initCar(time)

	EKF = EKFSLAM(car, landmarks)

	for t in range(1,time.shape[1]):
		tau = time[0,t] - time[0,t-1] 
    	# (a) IMU Localization via EKF Prediction
		EKF.Predict(linear_velocity[:, t], angular_velocity[:, t], tau, 0.00001, 0.0001)
		EKF.Update(features[:,:,t], K, b, imu_T_cam, 0.001)
        # record current poses
        #Car['trajectory'][:, :, i] = world_T_imu(Car['mean']) # inv(inv pose)
        #Landmarks['trajectory'][:, :, i - 1] = Landmarks['mean'][:]

        #Car['trajectory_vi'][:, :, i] = world_T_imu(Car['mean_vi']) 
        #Landmarks['trajectory_vi'][:, :, i - 1] = Landmarks['mean_vi'][:]

    	# (b) Landmark Mapping via EKF Update
        #EKF_visual_update(Car, Landmarks, features[:, :, i], K, b, cam_T_imu, 3500)
         
    	# (c) Visual-Inertial SLAM
        #EKF_visual_inertial_update(Car, Landmarks, features[:, :, i], K, b, cam_T_imu, 3500)
        
        # plotting
        #if ((i - 1) % 100 == 0 or i == t.shape[1] - 1):
        	# You can use the function below to visualize the robot pose over time
        #    visualize_trajectory_2d(Car['trajectory'], Landmarks['mean'], Car['trajectory_vi'], Landmarks['mean_vi'], timestamp = str(i), path_name = filename[7:-4], show_ori = True, show_grid = True)

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)
	visualize_trajectory_2d(world_T_imu, show_ori = True)

	print('a')
