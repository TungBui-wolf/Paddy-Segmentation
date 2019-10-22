import cv2
import numpy as np
import matplotlib.pyplot as plt
import pprint
import rasterio

a = cv2.imread('Output/predict_mask.tif')
a= a*255
plt.imshow(a)

 f = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

closing = cv2.morphologyEx(f, cv2.MORPH_CLOSE, kernel3)
for i in range(5):
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
del closing
for i in range(5):
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel2)
input_img = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel3)


plt.imshow(input_img)
# print(input_img)
# print(a.shape)

profiles = []
with rasterio.open('Input/img.tif') as src:
        profile = src.profile
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw')
with rasterio.open('Output/post-processed_predict_mask.tif','w',**profile) as dst:
        dst.write(input_img.astype(np.uint8), indexes=1)