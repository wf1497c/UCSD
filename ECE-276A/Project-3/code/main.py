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
	car['joint_covariance'] = 0.01 * np.eye(36) # (3M+6)*(3M+6), set M = 10
	car['trajectory_vi'] = np.zeros((4, 4, len(time[0]))) # running history

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
	landmarks['trajectory_vi'] = np.empty(features.shape)
	landmarks['covariance_vi'] = np.zeros((3, 3, num_features))
	for i in range(num_features):
		landmarks['covariance_vi'][:, :, i] = 0.01 * np.eye(3)

	return landmarks

def world_T_imu(mean_pose):
    R_T = np.transpose(mean_pose[:3, :3])
    p = mean_pose[:3, 3].reshape(3, 1)
    
    U_inv = np.vstack((np.hstack((R_T, -np.dot(R_T, p))), np.array([0, 0, 0, 1])))
    
    return U_inv

if __name__ == '__main__':
	# Load the measurements
	filename = "code/data/03.npz"
	time,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

	landmarks = initLandmarks(features)
	car = initCar(time)

	EKF = EKFSLAM(car, landmarks)

	for t in range(1,time.shape[1]):
		print('--------' + str(t+1) + 'step --------')
		tau = time[0,t] - time[0,t-1] 
    	# (a) IMU Localization via EKF Prediction
		EKF.inertialPrediction(t, linear_velocity[:, t], angular_velocity[:, t], tau, 0.001, 0.001)
		EKF.visualInertialPrediction(t, linear_velocity[:, t], angular_velocity[:, t], tau, 0.001, 0.001)
		
		car['trajectory_vi'][:, :, t] = car['mean_vi']
		landmarks['trajectory_vi'][:, :, t - 1] = landmarks['mean_vi'][:]

    	# (b) Landmark Mapping via EKF Update
		EKF.update(t, features[:,:,t], K, b, imu_T_cam, 0.001)
         
    	# (c) Visual-Inertial SLAM
		#EKF.visualInertialUpdate(t, features[:, :, t], K, b, imu_T_cam, 100)
        
        # plotting
		if (t % 50 == 0):
			filename = 'results/trajectory03.png'
			filename_gif = 'results/trajectory03.gif'
			#filename2 = 'results/trajectory10_vi'
			visualize_trajectory_2d(car['trajectory'], landmarks['trajectory'][:,:,t], filename, filename_gif, show_ori = True)
			#visualize_trajectory_2d(car['trajectory_vi'], landmarks['trajectory_vi'][:,:,t], filename2, show_ori = True)
