'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import pickle
from generate_rgb_data import read_pixels
from glob import glob
import math
import os

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    # Unpacked parameters from parameters.pkl
    params_path = os.path.abspath('pixel_classification/parameters.pkl')
    with open(params_path, 'rb') as f:
      params = pickle.load(f)

    self.mu_r, self.mu_g, self.mu_b = params[0], params[1], params[2]
    self.cov_r, self.cov_g, self.cov_b = params[3], params[4], params[5]
    self.prior_r, self.prior_g, self.prior_b = params[6], params[7], params[8]

  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Just a random classifier for now
    # Replace this with your own approach 
    
    y = np.empty(len(X))

    for i in range(X.shape[0]):
      x = X[i]
      log_likelihood_r = self.gaussian_class_conditional_prob(x, self.mu_r, self.cov_r, self.prior_r)
      log_likelihood_g = self.gaussian_class_conditional_prob(x, self.mu_g, self.cov_g, self.prior_g)
      log_likelihood_b = self.gaussian_class_conditional_prob(x, self.mu_b, self.cov_b, self.prior_b)
      likelihood = [log_likelihood_r, log_likelihood_g, log_likelihood_b]
      y[i] = 1 + np.argmin(likelihood)
    
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

  def gaussian_class_conditional_prob(self, x, mu, cov, prior):
    '''
      Use Baysian decision rule to compute P(data|class)
      Return class conditional probability of gaussian distribution prior
      i(x) = argmax(P(x|i)) where i is the predicted class and x is input data
    '''
    log_likelihood = (x - mu).T.dot(np.linalg.inv(cov)).dot(x - mu) + np.log((2*math.pi) ** 3) * np.linalg.det(cov) - 2 * np.log(prior)
    #log_likelihood = (x - mu).T.dot(np.linalg.inv(cov)).dot(x - mu) + np.log(np.linalg.det(cov)) - 2 * np.log(prior)
    return log_likelihood

if __name__ == '__main__':
  folder = 'pixel_classification/data/training'
  Xr = read_pixels(folder+'/red', verbose = True)
  Xg = read_pixels(folder+'/green')
  Xb = read_pixels(folder+'/blue')
  yr, yg, yb = np.full(Xr.shape[0],1), np.full(Xg.shape[0], 2), np.full(Xb.shape[0],3)
  X, y = np.concatenate((Xr,Xg,Xb)), np.concatenate((yr,yg,yb))

  pixel_classifier = PixelClassifier()
  y_pred = pixel_classifier.classify(X)
  acc = np.sum(y == y_pred) / len(y)
  print(acc)