## import packages
import numpy as np
import pandas as pd
from PIL import Image
from numpy import histogram as hist  # call hist, otherwise np.histogram
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module
import rpc_module



def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def grid_search(model_images, query_images, dist_type, hist_type, num_bins):

    try:
        [best_match, D] = match_module.find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    except:
        print(dist_type, hist_type, num_bins, 'errore linea 25')
        num_correct = 0
        recog_rate = 0

    try:
        num_correct = sum(best_match == range(len(query_images)))
        recog_rate = num_correct / len(query_images)
    except:
        print(dist_type, hist_type, num_bins, 'errore linea 30')

    return [[dist_type, hist_type, num_bins, num_correct, recog_rate]]

def grid_search_multiprocessing(model_images, query_images, dist_list, hist_list, num_bins):
    from multiprocessing import cpu_count
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from itertools import product

    ncpus = cpu_count()
    results = []

    grid = list(product(*[dist_list, hist_list, num_bins])) #zip(dist_list, hist_list, num_bins)

    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        futures = list((executor.submit(grid_search, model_images, query_images, trip[0], trip[1], trip[2] ) for trip in grid))

    for future in as_completed(futures):
        results += future.result()

    final_res = pd.DataFrame(results, columns=['dist_type', 'hist_type', 'num_bins', 'correct', 'recog_rate'])
    final_res.to_csv('grid_search.csv', index=False)

    return

## gray-value histograms (Question 2.a)
"""
img_color = np.array(Image.open('./model/obj100__0.png'))
img_gray = rgb2gray(img_color.astype('double'))

plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(img_color)

plt.subplot(1,3,2)
num_bins_gray = 40
hist_gray1, bin_gray1 = hist(img_gray.reshape(img_gray.size), num_bins_gray, (0,255))
plt.bar((bin_gray1[0:-1] + bin_gray1[1:])/2, hist_gray1)

plt.subplot(1,3,3)
hist_gray2, bin_gray2 = histogram_module.normalized_hist(img_gray, num_bins_gray)
plt.bar((bin_gray2[0:-1] + bin_gray2[1:])/2, hist_gray2)

plt.show()




## more histograms (Question 2.b)

#Compose and test RGB histograms (histogram_module.rgb_hist)
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(img_color)

num_bins_color = 5
plt.subplot(1,2,2)
hist_rgb = histogram_module.rgb_hist(img_color.astype('double'), num_bins_color)
plt.bar(np.array(range(1,hist_rgb.size+1)),hist_rgb)
plt.show()

#Compose and test RG histograms (histogram_module.rg_hist)
plt.figure(3)
plt.subplot(1,2,1)
plt.imshow(img_color)

num_bins_color = 5
plt.subplot(1,2,2)
hist_rg = histogram_module.rg_hist(img_color.astype('double'), num_bins_color)
plt.bar(np.array(range(1,hist_rg.size+1)),hist_rg)
plt.show()

#Compose and test dxdy histograms (histogram_module.dxdy_hist)
plt.figure(5)
plt.subplot(1,2,1)
plt.imshow(img_color, cmap='gray')

num_bins_dxdy = 10
plt.subplot(1,2,2)
hist_dxdy = histogram_module.dxdy_hist(img_gray, num_bins_dxdy)
plt.bar(np.array(range(1,hist_dxdy.size+1)),hist_dxdy)
plt.show()



## Distance functions (Question 2.c)

image_files1 = ['./model/obj1__0.png']
image_files2 = ['./model/obj91__0.png', './model/obj94__0.png']

plt.figure()
plt.subplot(1,3,1); plt.imshow(np.array(Image.open(image_files1[0])), vmin=0, vmax=255); plt.title(image_files1[0])
plt.subplot(1,3,2); plt.imshow(np.array(Image.open(image_files2[0])), vmin=0, vmax=255); plt.title(image_files2[0])
plt.subplot(1,3,3); plt.imshow(np.array(Image.open(image_files2[1])), vmin=0, vmax=255); plt.title(image_files2[1])
plt.show()

img1_color = np.array(Image.open(image_files1[0]))
img2_color = np.array(Image.open(image_files2[0]))
num_bins_color = 30
hist1_rgb = histogram_module.rgb_hist(img1_color.astype('double'), num_bins_color)
hist2_rgb = histogram_module.rgb_hist(img2_color.astype('double'), num_bins_color)

print('Distance according to dist_l2 %f'% dist_module.dist_l2(hist1_rgb,hist2_rgb))
print('Distance according to dist_interset %f'% dist_module.dist_intersect(hist1_rgb,hist2_rgb))
print('Distance according to dist_chi2 %f'% dist_module.dist_chi2(hist1_rgb,hist2_rgb))


distance_types = ['l2', 'intersect', 'chi2']
print('distance functions:', distance_types)

hist_types = ['grayvalue', 'rgb', 'rg', 'dxdy']
print('histogram types:', hist_types)

num_bins_color = 30;
num_bins_gray = 90;

for img1_file in image_files1:
    img1_color = np.array(Image.open(img1_file))
    img1_color = img1_color.astype('double')
    img1_gray = rgb2gray(img1_color)
    
    for img2_file in image_files2:
        img2_color = np.array(Image.open(img2_file))
        img2_color = img2_color.astype('double')
        img2_gray = rgb2gray(img2_color)
    
        D = np.zeros( (len(distance_types),len(hist_types)) )
    
        for didx in range(len(distance_types)):
    
            for hidx in range(len(hist_types)):
    
                if histogram_module.is_grayvalue_hist(hist_types[hidx]):
                    hist1 = histogram_module.get_hist_by_name(img1_gray, num_bins_gray, hist_types[hidx])
                    hist2 = histogram_module.get_hist_by_name(img2_gray, num_bins_gray, hist_types[hidx])
            
                else:
                    hist1 = histogram_module.get_hist_by_name(img1_color, num_bins_color, hist_types[hidx])
                    hist2 = histogram_module.get_hist_by_name(img2_color, num_bins_color, hist_types[hidx])
    
                if len(hist1) == 2:
                    hist1 = hist1[0]
                if len(hist2) == 2:
                    hist2 = hist2[0]
    
                D[didx, hidx] = dist_module.get_dist_by_name(hist1, hist2, distance_types[didx])
         
        print('Comparison of images "%s" to "%s":'% (img1_file, img2_file))
        print(D)
        print('\n')


print('Legend:')
print('%s-%s, %s-%s, %s-%s, %s-%s'%('l2', 'grayvalue', 'l2', 'rgb', 'l2', 'rg', 'l2', 'dxdy'))
print('%s-%s, %s-%s, %s-%s, %s-%s'%('intersect', 'grayvalue', 'intersect', 'rgb', 'intersect', 'rg', 'intersect', 'dxdy'))
print('%s-%s, %s-%s, %s-%s, %s-%s'%('chi2', 'grayvalue', 'chi2', 'rgb', 'chi2', 'rg', 'chi2', 'dxdy'))





"""
## Find best match (Question 3.a)

with open('model.txt') as fp:
    model_images = fp.readlines()
model_images = [x.strip() for x in model_images] 

with open('query.txt') as fp:
    query_images = fp.readlines()
query_images = [x.strip() for x in query_images] 

dist_type = 'intersect';
hist_type = 'rg';
num_bins = 30;

[best_match, D] = match_module.find_best_match(model_images, query_images, dist_type, hist_type, num_bins)




## visualize nearest neighbors (Question 3.b)
query_images_vis = [query_images[i] for i in np.array([0,4,9])]
match_module.show_neighbors(model_images, query_images_vis, dist_type, hist_type, num_bins)



## compute recognition percentage (Question 3.c)
# import ipdb; ipdb.set_trace()
num_correct = sum( best_match == range(len(query_images)) )
print('number of correct matches: %d (%f)\n'% (num_correct, 1.0 * num_correct / len(query_images)))

# decide whether to perform or not multiprocessing grid search
grid_boolean = True

if grid_boolean:
    print('performing grid search')
    dist_list = ['chi2', 'l2', 'intersect']
    hist_list = ['grayvalue', 'rgb', 'rg', 'dxdy']
    num_bins = [5]

    grid_search_multiprocessing(model_images, query_images, dist_list, hist_list, num_bins)

results = pd.read_csv('grid_search.csv')
print(results.sort_values(by=['recog_rate'], ascending=False).head())

"""
## plot recall_precision curves (Question 4)

with open('model.txt') as fp:
    model_images = fp.readlines()
model_images = [x.strip() for x in model_images] 

with open('query.txt') as fp:
    query_images = fp.readlines()
query_images = [x.strip() for x in query_images] 

num_bins = 20;


plt.figure(8)
rpc_module.compare_dist_rpc(model_images, query_images, ['chi2', 'intersect', 'l2'], 'rg', num_bins, ['r', 'g', 'b'])
plt.title('RG histograms')
plt.show()


plt.figure(9)
rpc_module.compare_dist_rpc(model_images, query_images, ['chi2', 'intersect', 'l2'], 'rgb', num_bins // 2, ['r', 'g', 'b'])
plt.title('RGB histograms')
plt.show()


plt.figure(10)
rpc_module.compare_dist_rpc(model_images, query_images, ['chi2', 'intersect', 'l2'], 'dxdy', num_bins, ['r', 'g', 'b'])
plt.title('dx/dy histograms')
plt.show()
"""