# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):

    low = int(-3*sigma)
    high = int(3*sigma)

    x = np.arange(low, high + 1, 1)
    Gx = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-x**2/(2*sigma**2))

    return Gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    
    sigma = int(sigma)

    Gx, x = gauss(sigma)
    delta_idx = 3*sigma

    padded_img = np.pad(img, (3*sigma, 3*sigma), 'constant')
	
    smooth_img = np.zeros((img.shape[0], img.shape[1]))


    # Ix
    for m in range(delta_idx, padded_img.shape[0] - delta_idx):
    	for n in range(delta_idx, padded_img.shape[1] - delta_idx):
    		smooth_img[m - delta_idx, n - delta_idx] = np.dot(padded_img[m, n-delta_idx:n+delta_idx + 1], Gx)

    padded_img = np.pad(smooth_img, (3*sigma, 3*sigma), 'constant')

    # Ix
    for m in range(delta_idx, padded_img.shape[0] - delta_idx):
    	for n in range(delta_idx, padded_img.shape[1] - delta_idx):
    		smooth_img[m - delta_idx, n - delta_idx] = np.dot(padded_img[m - delta_idx:m+delta_idx+1, n], Gx)


    return smooth_img

"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    low = int(-3*sigma)
    high = int(3*sigma)

    x = np.arange(low, high + 1, 1)
    Dx = -(1/np.sqrt(2*np.pi*sigma**3)*x*np.exp(-x**2/(2*sigma**2)))
    
    return Dx, x



def gaussderiv(img, sigma):

        
    sigma = int(sigma)

    Dx, x = gaussdx(sigma)
    delta_idx = 3*sigma

    padded_img = np.pad(img, (3*sigma, 3*sigma), 'constant')
    
    imgDx = np.zeros((img.shape[0], img.shape[1]))
    imgDy = np.zeros((img.shape[0], img.shape[1]))


    # Ix
    for m in range(delta_idx, padded_img.shape[0] - delta_idx):
        for n in range(delta_idx, padded_img.shape[1] - delta_idx):
            imgDx[m - delta_idx, n - delta_idx] = np.dot(padded_img[m, n-delta_idx:n+delta_idx + 1], Dx)


    # Ix
    for m in range(delta_idx, padded_img.shape[0] - delta_idx):
        for n in range(delta_idx, padded_img.shape[1] - delta_idx):
            imgDy[m - delta_idx, n - delta_idx] = np.dot(padded_img[m - delta_idx:m+delta_idx+1, n], Dx)
    
    return imgDx, imgDy

