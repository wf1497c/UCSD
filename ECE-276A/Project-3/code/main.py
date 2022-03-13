import numpy as np
from pr3_utils import *
from scipy.linalg import expm
from src.EKF import EKFSLAM

def initCar(time):
	car = {}
	car['mean'] = np.eye(4) # in SE(3)
	car['covariance'] = 0.01 * np.eye(6)
	car['trajectory'] = np.zeros((4, 4, len(time[0]))) # running history

	# For visual-inertial SLAM
	car['mean_vi'] = np.eye(4) # in SE(3)
	car['covariance_vi'] = 0.01 * np.eye(6)

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
		
	# For visual-inertial SLAM
	landmarks['mean_vi'] = np.empty(features.shape[:2])
	landmarks['mean_vi'].fill(np.nan)
	landmarks['covariance_vi'] = np.zeros((3, 3, num_features))
	for i in range(num_features):
		landmarks['covariance_vi'][:, :, i] = 0.01 * np.eye(3)

	return landmarks


if __name__ == '__main__':
	# Load the measurements
	filename = "code/data/10.npz"
	time,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

	landmarks = initLandmarks(features)
	car = initCar(time)

	EKF = EKFSLAM(car, landmarks)

	for t in range(1,time.shape[1]):
		print('--------' + str(t+1) + 'step --------')
		tau = time[0,t] - time[0,t-1] 
    	# (a) IMU Localization via EKF Prediction
		EKF.prediction(t, linear_velocity[:, t], angular_velocity[:, t], tau, 0.00001, 0.0001)

    	# (b) Landmark Mapping via EKF Update
		EKF.update(t, features[:,:,t], K, b, imu_T_cam, 0.001)
         
    	# (c) Visual-Inertial SLAM
		EKF.visual_inertial_update(t, features[:, :, t], K, b, imu_T_cam, 3500)
        
        # plotting
		if (t % 100 == 0):
			visualize_trajectory_2d(car['trajectory'], landmarks['trajectory'][:,:,t], show_ori = True)
