'''
: Get parameters for gaussian classifer: mean, covariance, and prior probability
'''

import numpy as np
from glob import glob
import pickle

class GaussianClassifierParameters():
    def __init__(self, x_pos, x_neg):
        self.pos = x_pos
        self.neg = x_neg
        self.total_num = x_pos.shape[0] + x_neg.shape[0]

    def prior(self, x):
        '''
        : prior is defined as ratio of each class
        '''
        return x.shape[0] / self.total_num

    def mean(self, x):
        '''
        : class mean
        '''
        return np.mean(x, axis = 0)
    
    def covariance(self, x):
        '''
        : class covariance
        '''
        return np.cov(x.T)

    def package_gc_params(self):
        '''
        : get parameters for gaussian classifier: covariance and prior of each class
        '''
        self.mu_pos = self.mean(self.pos)
        self.mu_neg = self.mean(self.neg)
        self.cov_pos = self.covariance(self.pos)
        self.cov_neg = self.covariance(self.neg)
        self.prior_pos = self.prior(self.pos)
        self.prior_neg = self.prior(self.neg)
        params = [self.mu_pos, self.mu_neg, self.cov_pos, self.cov_neg, self.prior_pos, self.prior_neg]
        with open('parameters_bin_detector.pkl', 'wb') as f:
            pickle.dump(params, f)

        return params

def valid_data(x):
    valid_data_list = []
    for pixel in x:
        if(max(pixel) <= 255):
            pixel = [p / 255 for p in pixel]
            valid_data_list.append(pixel)

    return np.array(valid_data_list, dtype = np.float)

if __name__ == '__main__':

    with open('full_yuv_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_pos, X_neg = data
    pos = valid_data(X_pos)
    neg = valid_data(X_neg)

    gcp = GaussianClassifierParameters(pos, neg)
    print(gcp.package_gc_params())

print('a')