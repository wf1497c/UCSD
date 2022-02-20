'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''
import numpy as np
import cv2
from skimage.measure import label, regionprops
import pickle
import math
import matplotlib.pyplot as plt


class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		with open('parameters_bin_detector.pkl', 'rb') as f:
			params = pickle.load(f)

		self.mu_pos, self.mu_neg, self.cov_pos, self.cov_neg, self.prior_pos, self.prior_neg = params

		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		img = img / 255
		mask = img[:,:,0] # one channel for binary image
		
		for r in range(img.shape[0]):
			for c in range(img.shape[1]):
				pixel = img[r,c,:]
				# log-likelihood of classes
				log_likelihood_pos = self.gaussian_class_conditional_prob(pixel, self.mu_pos, self.cov_pos, self.prior_pos)
				log_likelihood_neg = self.gaussian_class_conditional_prob(pixel, self.mu_neg, self.cov_neg, self.prior_neg)
				if(log_likelihood_pos <= log_likelihood_neg):
					mask[r,c] = 1
				else:
					mask[r,c] = 0
		
		return mask

		# YOUR CODE BEFORE THIS LINE
		################################################################

	def get_bounding_boxes(self, mask_img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 
		mask = mask_img
		mask = mask * 255
		x_max, y_max = mask.shape[0], mask.shape[1]	
		
		# Erosion with dilation "expands" the whole area of bins 
		kernel = np.ones((13,13), np.uint8)
		mask_erode = cv2.erode(mask, kernel, iterations = 1)
		mask_dilation = cv2.dilate(mask_erode, kernel[:5,:5], iterations = 3)

		boxes = []
		similarity = []
		contours, _ = cv2.findContours(mask_dilation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		for cnt in contours:
			# Draw bounding box 
			x,y,w,h = cv2.boundingRect(cnt)
			area_ratio = cv2.contourArea(cnt)/(y_max*x_max)
			similarity1 = 100 - np.absolute((h/w)-1.5)*100
			
			# Choose bounding box with appropriate size
			if 0.7 <= h/w <=2 and area_ratio > 0.02:
				similarity1 = 100 - np.absolute((h/w)-1.5)*100
				
				boxes.append([x,y,x + w,y + h])
				
				similarity.append(similarity1)
		boxes.sort()

		return boxes
	
	def gaussian_class_conditional_prob(self, x, mu, cov, prior):
		'''
		Use Baysian decision rule to compute P(data|class)
		Return class conditional probability of gaussian distribution prior
		i(x) = argmax(P(x|i)) where i is the predicted class and x is input data
		'''
		log_likelihood = (x - mu).T.dot(np.linalg.inv(cov)).dot(x - mu) + np.log((2*math.pi) ** 3) * np.linalg.det(cov) - 2 * np.log(prior)
		return log_likelihood


if __name__ == '__main__':

	folder = 'bin_detection/data/validation/'
	file = folder + '0061.jpg'
	img = cv2.imread(file)

	binDetector = BinDetector()
	mask = binDetector.segment_image(img)
	print(mask.shape)
	plt.imshow(mask, cmap = 'gray')
	plt.savefig('mask.png')   

	boxes = binDetector.get_bounding_boxes(mask)
	for box in boxes: 
		x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
		cv2.rectangle(img, (x1,y1), (x2, y2), (0,0,255), 5)
		fig,ax = plt.subplots(figsize = (8,6))
		ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		ax.axis('off')
		plt.savefig('boxes.png')

	print('a')


