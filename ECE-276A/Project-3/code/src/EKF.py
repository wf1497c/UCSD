from matplotlib.pyplot import sci
import numpy as np
from scipy.linalg import expm

def hatMapping(vec):
    '''
    Input: 
        vec: 3d vector
    Ouput:
        vec_hat: corresponding skew sym. matrix
    '''
    vec_hat = np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])
    
    return vec_hat

def hatMapping_6(vec):
    theta = vec[3:, np.newaxis]
    p = vec[:3, np.newaxis]
    
    hat_map = np.block([[hatMapping(theta), -p],
                         [np.zeros((1, 4))]])
    
    return hat_map

def cameraCalibrationMatrix(K, b):
    M = np.hstack((np.vstack((K[:2,:],K[:2,:])),np.zeros([4,1])))
    M[2,3] = -K[0, 0] * b

    return M

def reverseTransformation(T):
    R = T[0:3,0:3]
    p = np.array([T[:3, 3],]).T
    T_inv = np.vstack((np.hstack((R.T, -R.T @ p)), np.zeros([1,4])))
    T_inv[3,3] = 1

    return T_inv

def projection(q):
    return q / q[2]

def projection_derivative(q):
    derivative = np.array([[1, 0, -q[0]/q[2], 0],
                           [0, 1, -q[1]/q[2], 0],
                           [0, 0,          0, 0],
                           [0, 0, -q[3]/q[2], 1]])
    return derivative / q[2]

def se3Mapping(vec):
    s = vec[:3]
    a = hatMapping(vec)
    res = np.vstack((np.hstack((np.eye(3), -hatMapping(s))), np.zeros((1,6))))

    return res

def buildCovBlocks(covs,B):
    nr = covs.shape[2]
    nc = covs.shape[2]
    m, n = (3,3)

    A = np.zeros([m*nr,n*nc])
    C = np.zeros([m*nr,6])
    for i in range(nr):
        cov = covs[:,:,i]
        A = A + np.kron(np.eye(nr,nc,-i),cov)
    A = np.hstack((A, C))
    B = np.hstack((C.T,B))
    A = np.vstack((A,B))

    return A

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

class EKFSLAM():
    def __init__(self, car, landmarks) -> None:
        self.car = car
        self.landmarks = landmarks

        pass
    
    def inertialPrediction(self, t, v, w, tau, weight_v, weight_w):
        '''
        Input:
            v: 3 X 1 linear velocity
            w: 3 X 1 angular velocity
            sigma_w: variance of noise in motion model
        '''
        car = self.car
        w_hat = hatMapping(w)
        v_hat = hatMapping(v)
        u_hat = np.vstack((np.hstack((w_hat, np.array([v,]).T)), np.zeros([1,4])))
        u_curly_hat = np.block([
            [w_hat,           v_hat],
            [np.zeros([3,3]), w_hat]
        ])
        W = np.block([
            [weight_v * np.eye(3), np.zeros([3,3])],
            [np.zeros([3,3]), weight_w * np.eye(3)]
        ])
        
        car['mean'] = car['mean'] @ expm(tau * u_hat) # in SE(3)
        car['covariance'] = expm(-tau * u_curly_hat) @ car['covariance'] @ expm(-tau * u_curly_hat).T + W

        self.car = car
    
    def visualInertialPrediction(self, t, v, w, tau, weight_v, weight_w):
        '''
        Input:
            v: 3 X 1 linear velocity
            w: 3 X 1 angular velocity
            sigma_w: variance of noise in motion model
        '''
        car = self.car
        w_hat = hatMapping(w)
        v_hat = hatMapping(v)
        u_hat = np.vstack((np.hstack((w_hat, np.array([v,]).T)), np.zeros([1,4])))
        u_curly_hat = np.block([
            [w_hat,           v_hat],
            [np.zeros([3,3]), w_hat]
        ])
        W = np.block([
            [weight_v * np.eye(3), np.zeros([3,3])],
            [np.zeros([3,3]), weight_w * np.eye(3)]
        ])
        
        car['mean_vi'] = car['mean_vi'] @ expm(tau * u_hat) # in SE(3)
        car['covariance_vi'] = expm(-tau * u_curly_hat) @ car['covariance_vi'] @ expm(-tau * u_curly_hat).T + W
        car['trajectory'][:, :, t] = car['mean'] # inv(inv pose)
        self.car = car

    def update(self, t, cur_features, K, b, imu_T_cam, weight_v = 1000):
        V = np.eye(4) * weight_v
        M = cameraCalibrationMatrix(K, b)
        P = np.eye(3, 4)
        Landmarks = self.landmarks
        car = self.car

        # camera
        w_T_cam = car['mean'] @ imu_T_cam

        for i in range(cur_features.shape[1]):
            z = cur_features[:,i]
            d = cur_features[0,i] - cur_features[2,i]
            if(d == 0):
                continue
            z0 = K[0, 0] * b / d

            if (np.all(z == -1)):
                continue

            if (np.all(np.isnan(Landmarks['mean'][:, i]))):
                #camera_frame_coords = z0 * np.linalg.inv(M) @ np.array([[z],]).T
                x_cam = z0 * (z[0] - M[0,2]) / M[0,0]
                y_cam = z0 * (z[1] - M[1,2]) / M[1,1]
                landmark_cam = np.array([[x_cam, y_cam, z0, 1]]).T
                landmark_world = w_T_cam @ landmark_cam
                Landmarks['mean'][:, i] = landmark_world[:,0]

                continue
            
            cam_T_w = reverseTransformation(w_T_cam)
            
            landmark_cam = cam_T_w @ Landmarks['mean'][:, i]

            z_tilde = M @ projection(landmark_cam)
            H = M @ projection_derivative(landmark_cam) @ cam_T_w @ P.T

            # perform the visual EKF update
            Kalman_gain = Landmarks['covariance'][:, :, i] @ H.T @ np.linalg.inv(H @ Landmarks['covariance'][:, :, i] @ H.T + V)
            Landmarks['mean'][:, i] = Landmarks['mean'][:, i] + P.T @ Kalman_gain @ (z - z_tilde)
            Landmarks['covariance'][:, :, i] = (np.eye(3) - Kalman_gain @ H) @ Landmarks['covariance'][:, :, i]
        
        Landmarks['trajectory'][:, :, t] = Landmarks['mean'][:]
        self.landmarks = Landmarks
            #self.car = car
    
    def visualInertialUpdate(self, t, cur_features, K, b, imu_T_cam, weight_v):
        V = weight_v * np.eye(4)
        P = np.eye(3, 4)
        M = cameraCalibrationMatrix(K, b)
        Landmarks = self.landmarks
        car = self.car
        # camera
        w_T_cam = car['mean_vi'] @ imu_T_cam
        cam_T_w = reverseTransformation(w_T_cam)
        landmarks_cam_list = []
        
        for i in range(cur_features.shape[1]):
            z = cur_features[:,i]
            d = cur_features[0,i] - cur_features[2,i]
            if(d == 0):
                continue
            z0 = K[0, 0] * b / d
            z = cur_features[:, i][:]
            
            if (np.all(z == -1)):
                continue

            if (np.all(np.isnan(Landmarks['mean_vi'][:, i]))):
                x_cam = z0 * (z[0] - M[0,2]) / M[0,0]
                y_cam = z0 * (z[1] - M[1,2]) / M[1,1]
                landmark_cam = np.array([[x_cam, y_cam, z0, 1]]).T
                landmarks_cam_list.append(landmark_cam)
                landmark_world = w_T_cam @ landmark_cam
                Landmarks['mean_vi'][:, i] = landmark_world[:,0]
                H = M @ projection_derivative(landmark_cam) @ cam_T_w @ P.T

                continue
        
        landmarks_cam_list = np.array(landmarks_cam_list)
        indvalid = np.isnan(Landmarks['mean_vi'][0,:]) == False    
        validlandmarks = Landmarks['mean_vi'][:,indvalid]
        validCovBlocks = Landmarks['covariance'][:,:,indvalid]
        covBlocks = buildCovBlocks(validCovBlocks, car['covariance']) # 3M+6 by 3M+6
        H = M @ projection_derivative(landmark_cam) @ cam_T_w @ P.T

        #jointCov = np.block([])
            #if np.isnan(Landmarks['mean_vi'][0, i]) == False:
            #    cam_T_w = reverseTransformation(w_T_cam)
            #    landmark_cam = cam_T_w @ Landmarks['mean_vi'][:, i]
            #    z_tilde = M @ projection(landmark_cam) 
            #    H = M @ projection_derivative(landmark_cam) @ cam_T_w @ P.T

                # perform visual EKF update
            #    Kalman_gain = Landmarks['covariance_vi'][:, :, i] @ H.T @ np.linalg.inv(H @ Landmarks['covariance_vi'][:, :, i] @ H.T + V)
            #    Landmarks['mean_vi'][:, i] = Landmarks['mean_vi'][:, i] + P.T @ Kalman_gain @ (z - z_tilde)
            #    Landmarks['covariance_vi'][:, :, i] = (np.eye(3) - Kalman_gain @ H) @ Landmarks['covariance_vi'][:, :, i]
                
            #    cam_T_imu = reverseTransformation(imu_T_cam)
            #    curr_landmark = car['mean_vi'] @ Landmarks['mean_vi'][:, i] # in IMU frame
                # form H; the Jacobian of z_tilde w.r.t. current car's inverse pose evaluated at car's current position
            #    H = M @ projection_derivative(cam_T_imu @ curr_landmark) @ cam_T_imu @ np.block([[np.eye(3), -hatMapping(curr_landmark[:3])],
                                                                                                #[np.zeros((1, 6))]])
                # perform the inertial EKF update
            #    KG = car['covariance_vi'] @ H.T @ np.linalg.inv(H @ car['covariance_vi'] @ H.T + V)
                
            #    car['mean_vi'] = expm(hatMapping_6(KG @ (z - z_tilde))) @ car['mean_vi']
            #    car['covariance_vi'] = (np.eye(6) - KG @ H) @ car['covariance_vi']
        # Landmarks['trajectory_vi'][:,:,t] = Landmarks['mean_vi']
        # self.landmarks = Landmarks
