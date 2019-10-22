import fiona
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import pprint

src1 = []
src2 = []
with fiona.open("Input/Shape file /paddy_bg.shp", "r") as shapefile:
    for i in shapefile:
        pprint.pprint(i)

    features = [f["properties"] for f in shapefile]
    for i in range(len(features)):
      if(features[i]['class'] == 'paddy'):
        src1.append(shapefile[i])
      elif(features[i]['class'] == 'background'):
        src2.append(shapefile[i])
       
    # pprint.pprint(src1)

    geo_paddy = [geo_paddy["geometry"] for geo_paddy in src1]
    geo_background = [geo_background["geometry"] for geo_background in src2]
    # pprint.pprint(geo_paddy)
     
pprint.pprint(features)

    with rasterio.open("Input/img.tif") as src:
    height = src.height
    width = src.width
    src_transform = src.transform
    out_meta = src.meta.copy()
    out_meta.update({'count':1})
mask_paddy = rasterio.features.geometry_mask(geo_paddy, (height, width), src_transform,invert=True, all_touched=True).astype(np.uint32)
mask_background = rasterio.features.geometry_mask(geo_background, (height, width), src_transform,invert=True, all_touched=True).astype(np.uint32)

plt.imshow(mask_paddy)
# plt.imshow(mask_background)

mask = 1*mask_paddy+2*mask_background
print(mask.shape)
plt.imshow(mask)
with rasterio.open("Input/mask.tif", "w", **out_meta) as dest:
    dest.write(mask,indexes=1)
    print(dest.count)

mask = mask.flatten().reshape(-1,1)
print(len(mask))

Img_pixels = []
with rasterio.open("Input/img.tif") as src:
  Img_pixels.append(src.read())

Img_pixels = np.array(Img_pixels)
print(Img_pixels)



Img_pixels = np.reshape(Img_pixels,(4,height,width))
print(Img_pixels.shape)
print(Img_pixels[0].shape)

pixel_bancolor = [Img_pixels[i].flatten() for i in range(4)]
pixel_bancolor = np.array(pixel_bancolor).T
print(pixel_bancolor)

pixel_color_padding=np.hstack((pixel_bancolor,mask))
print(pixel_color_padding)

labled_pixels = [pixel_color_padding[i,:] for i in range(mask.shape[0]) if(mask[i] > 0)]
labled_pixels = np.array(labled_pixels)
print(labled_pixels)

df = pd.DataFrame(labled_pixels, columns = ['ban1', 'ban2', 'ban3', 'ban4', 'padding'])
print(df.shape)
df.to_csv('Input/input_model.csv')