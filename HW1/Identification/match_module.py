import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
        
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(model_images), len(query_images)))
    
    
    if dist_type == "l2":
        fun_dist = dist_module.dist_l2
        fun_detect = np.argmin
    elif dist_type == "intersect":
        fun_dist = dist_module.dist_intersect
        fun_detect = np.argmax
    elif dist_type == "chi2":
        fun_dist = dist_module.dist_chi2
        fun_detect = np.argmin

    for i in range(len(model_hists)):
        for j in range(len(query_hists)):
            D[i,j] = fun_dist(model_hists[i], query_hists[j])
            
    best_match = fun_detect(D, axis = 1)
            
    return best_match, D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist
    
    if hist_type == "grayvalue":
        fun = histogram_module.normalized_hist
    elif hist_type == "rgb":
        fun = histogram_module.rgb_hist
    elif hist_type == "rg":
        fun = histogram_module.rg_hist
    elif hist_type == "dxdy":
        fun = histogram_module.dxdy_hist
    
    for i in range(len(image_list)):
        img = np.array(Image.open("./" + image_list[i]))
        img = img.astype('double')
        if hist_isgray:
            img = rgb2gray(img)
        image_hist.append(fun(img, num_bins))

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    #... (your code here)

