import os
import rasterio
import numpy as np
import csv
import pandas as pd
from keras.models import load_model

Img_pixels = []
with rasterio.open("Input/img.tif") as src:
  Img_pixels.append(src.read())

Img_pixels = np.array(Img_pixels)
print(Img_pixels)
Img_pixels = np.reshape(Img_pixels,(4,src.height,src.width))
print(Img_pixels.shape)
print(Img_pixels[0].shape)

pixel_bancolor = [Img_pixels[i].flatten() for i in range(4)]
pixel_bancolor = np.array(pixel_bancolor).T
print(pixel_bancolor)
print(len(pixel_bancolor))

model = load_model('Model/model_weight_paddy.h5')

p = model.predict(pixel_bancolor)
print(p)
p = (p <= 0.9).astype(np.uint8)
print(p)
print(p.shape)
p = np.argmax(p, axis=-1)
print(p)

import matplotlib.pyplot as plt

predict = p.reshape(608,552,1).transpose(2,0,1)
print(predict.shape)
plt.imshow(predict[0])

profile = []
with rasterio.open('Input/img.tif') as src:
        profile = src.profile
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw')
with rasterio.open('Output/predict_mask.tif','w',**profile) as dst:
  dst.write(predict[0].astype(np.uint8), indexes=1)
        