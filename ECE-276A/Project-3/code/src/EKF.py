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

class EKFSLAM():
    def __init__(self, car, landmarks) -> None:
        self.car = car
        self.landmarks = landmarks

        pass
    
    def Predict(self, v, w, tau, weight_v, weight_w):
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

    def Update(self, cur_features, K, b, imu_T_cam, weight_v = 1000):
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

            z_tilde = M @ projection(landmark_cam) # remove depth information via projection, and project to pixels
            # form H; the Jacobian of z_tilde w.r.t. current feature m evaluated at car's current position
            H = M @ projection_derivative(landmark_cam) @ cam_T_w @ P.T
            # perform the visual EKF update
            Kalman_gain = Landmarks['covariance'][:, :, i] @ H.T @ np.linalg.inv(H @ Landmarks['covariance'][:, :, i] @ H.T + V)
            
            Landmarks['mean'][:, i] = Landmarks['mean'][:, i] + P.T @ Kalman_gain @ (z - z_tilde)
            Landmarks['covariance'][:, :, i] = (np.eye(3) - Kalman_gain @ H) @ Landmarks['covariance'][:, :, i]

            #self.car = car
