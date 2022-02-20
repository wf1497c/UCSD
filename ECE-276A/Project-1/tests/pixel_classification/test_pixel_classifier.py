'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

from __future__ import division

from generate_rgb_data import read_pixels
from pixel_classifier import PixelClassifier
import numpy as np

if __name__ == '__main__':
  # test the classifier with ALL colors
  folder = 'pixel_classification/data/validation'#/blue'
  Xr = read_pixels(folder+'/red', verbose = True)
  Xg = read_pixels(folder+'/green')
  Xb = read_pixels(folder+'/blue')
  yr, yg, yb = np.full(Xr.shape[0],1), np.full(Xg.shape[0], 2), np.full(Xb.shape[0],3)
  X, y = np.concatenate((Xr,Xg,Xb)), np.concatenate((yr,yg,yb))
  
  #X = read_pixels(folder)
  myPixelClassifier = PixelClassifier()
  y_pred = myPixelClassifier.classify(X)
  
  print('Precision: %f' % (sum(y_pred == y)/y.shape[0]))

  
