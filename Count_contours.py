#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 14:47:30 2022

@author: rahulsingh
"""
import os 
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize


image_dir = '/home/rahulsingh/Downloads/data/larvaTest/label/prediction/'

image = Image.open(image_dir+'190.jpg_unet.jpeg') # Eventually can read all of them

new_image = image.resize((512, 288))
new_image.save(image_dir+'resized.jpeg')
img = cv2.imread(image_dir+'resized.jpeg',cv2.IMREAD_GRAYSCALE)

thresh = 200
#get threshold image
ret,thresh_img = cv2.threshold(img, thresh, 1, cv2.THRESH_BINARY)

plt.figure(1)
plt.imshow(thresh_img)
#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = np.zeros(img.shape)
image_copy = img.copy()
# draw the contours on the empty image
color = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)
cv2.drawContours(color, contours,-1, (0, 0, 255), 1, cv2.LINE_AA)
cv2.imshow("WTF",color)
#cv2.drawContours(image_copy, contours,11, (100,), -1)
contour_center = np.zeros((len(contours)))
contour_area = np.zeros((len(contours)))
cont_circum = np.zeros((len(contours)))
matches = np.zeros((len(contours)))
for i in range(len(contours)):
    contour_center = cv2.moments(contours[i])
    #contour_center[i] = cv2.moments(contours[i])[0]
    contour_area[i] = cv2.contourArea(contours[i])
    cont_circum[i] = cv2.arcLength(contours[i],True)
    matches[i] = cv2.matchShapes(contours[0], contours[i], 1, 0);
#cv2.imwrite('WTF', image_dir+'image_copy.jpeg')
#cv2.imwrite('D:/contours.png',img_contours) 
    
median_size = np.median(contour_area)
print(median_size)
part_larvaes = 0
overlapped_larvaes = 0
overlapped_contours=[]
artifacts=0
for i in range(len(contours)):
    if contour_area[i] < 0.5*median_size and contour_area[i] > 0.2*median_size:
        part_larvaes = part_larvaes+1
    if contour_area[i] > 1.0*median_size:
        overlapped_contours.append(contours[i])
        overlapped_larvaes = overlapped_larvaes+1
    if contour_area[i] < 0.2*median_size:
        artifacts=artifacts+1
print ("Overlapped Larvae", overlapped_larvaes)
print ("Part larve",part_larvaes)
print ("artifacts",artifacts)
print ("Total larva counts",len(contours)+overlapped_larvaes-artifacts)


skeleton = skeletonize(thresh_img)
plt.imshow(skeleton)

image = np.zeros(img.shape)
cv2.drawContours(image,overlapped_contours,-1, (255), -1)
cv2.imshow("WTF",image)
image=image.astype(int)
distance = ndi.distance_transform_edt(image)
distance1 = ndi.distance_transform_edt(distance)
coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=image)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = 100
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=4, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance1, cmap=plt.cm.gray)
ax[1].set_title('Distances')
#ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
#ax[2].set_title('Separated objects')

ax[2].imshow(mask, cmap=plt.cm.gray)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
