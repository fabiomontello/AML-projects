import numpy as np
from numpy import histogram as hist
from collections import Counter      # our import (for normalized_hist fun)



#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    bin_size = 255.0 / num_bins
    n_pixels = img_gray.size

    img_gray = (img_gray.flatten() // bin_size).astype(int)
    
    # creating the histogram and bins array using frequencies (Counter())
    empty_dict = Counter(list(range(num_bins+1)))
    freq = empty_dict + Counter(img_gray)
    freq.subtract(empty_dict)

    # taking hist and bins
    hists = np.array(list(freq.values())) / n_pixels # normalizing by n_pixels
    bins = np.array(list(freq.keys())) * bin_size    # obtaining float bin values 

    return hists[0:-1], bins



#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    bin_size = 255.0 / num_bins

    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))

    r_ch =  np.floor(np.array(img_color_double[:,:,0]).flatten()/bin_size).astype(np.int8)
    b_ch =  np.floor(np.array(img_color_double[:,:,2]).flatten()/bin_size).astype(np.int8)
    g_ch =  np.floor(np.array(img_color_double[:,:,1]).flatten()/bin_size).astype(np.int8)

    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):

        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        hists[r_ch[i], b_ch[i], g_ch[i]] += 1

        pass


    #Normalize the histogram such that its integral (sum) is equal 1
    n_pixels = img_color_double.shape[0]*img_color_double.shape[1]

    hists = hists / n_pixels

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    bin_size = 255.0 / num_bins

    r_ch =  np.floor(np.array(img_color_double[:,:,0]).flatten()/bin_size).astype(np.int8)
    g_ch =  np.floor(np.array(img_color_double[:,:,1]).flatten()/bin_size).astype(np.int8)

    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):

        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        hists[r_ch[i], g_ch[i]] += 1

        pass

    #Normalize the histogram such that its integral (sum) is equal 1
    n_pixels = img_color_double.shape[0]*img_color_double.shape[1]

    hists = hists / n_pixels

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists




#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    
    bin_size = 255.0 / num_bins

    Dx, Dy = gauss_module.gaussderiv(img_gray, 3, [-6,6])

    Dx = np.floor(Dx.flatten()/bin_size).astype(np.int8)
    Dy = np.floor(Dy.flatten()/bin_size).astype(np.int8)

    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))


    # Loop for each pixel i in the image 
    for i in range(img_gray.shape[0]*img_gray.shape[1]):

        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        hists[Dx[i], Dy[i]] += 1

        pass

    #Normalize the histogram such that its integral (sum) is equal 1
    n_pixels = img_gray.shape[0]*img_gray.shape[1]

    hists = hists / n_pixels

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name

