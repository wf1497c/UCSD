'''
: Get parameters for gaussian classifer: mean, covariance, and prior probability
'''

import numpy as np
import pickle
from generate_rgb_data import read_pixels
from glob import glob

class GaussianClassifierParameters():
    def __init__(self, red, green, blue):
        self.red, self.green, self.blue = red, green, blue
        self.total = len(self.red) + len(self.green) + len(self.blue)

    def prior(self, x):
        '''
        : Prior is defined as ration of data belonged to each class to number of all data
        '''
        return len(x) / self.total

    def covariance(self, x):
        '''
        : Covariance of all data in the same label
        '''
        return np.cov(x.T)

    def mean(self, x):
        '''
        : Mean value of all data in the same label
        '''
        return np.mean(x, axis = 0)

    def package_gc_params(self):
        mu_r = self.mean(self.red)
        mu_g = self.mean(self.green)
        mu_b = self.mean(self.blue)
        cov_r = self.covariance(self.red)
        cov_g = self.covariance(self.green)
        cov_b = self.covariance(self.blue)
        prior_r = self.prior(self.red)
        prior_g = self.prior(self.green)
        prior_b = self.prior(self.blue)
        params = [mu_r, mu_g, mu_b, cov_r, cov_g, cov_b, prior_r, prior_g, prior_b]
        with open('parameters.pkl', 'wb') as f:
            pickle.dump(params, f)

        return params


if __name__ == '__main__':

    folder = 'pixel_classification/data/training' 
    Xr = read_pixels(folder+'/red', verbose = False)
    Xg = read_pixels(folder+'/green')
    Xb = read_pixels(folder+'/blue')

    gcp = GaussianClassifierParameters(Xr, Xg, Xb)
    print(gcp.package_gc_params())