import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
import matplotlib.pyplot as plt

from glob import glob
import os
import gdal
import osgeo.osr as osr
from shutil import copy
import cv2
from collections import OrderedDict 
from PIL import Image
import numpy as np
from tqdm import tqdm


path = "/scratch/eva/TENCENT/Inria_dataset/AerialImageDataset/train/images"
img_list = glob(os.path.join(path,"*.tif"))


render_ds_path = "./osm_renders_0.3"
os.makedirs(render_ds_path, exist_ok=True)

miss_images = []
for img_path in tqdm(img_list):
    # open the dataset and get the geo transform matrix
    ds = gdal.Open(img_path) 
    # GDAL affine transform parameters, According to gdal documentation
    # xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
    xoff, a, b, yoff, d, e = ds.GetGeoTransform()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    shape = [(0,0), (0, w), (h, w) , (h, 0)]
    transformed_shape = []
    geotransformed_shape = []


    for (x,y) in shape:
        # supposing x and y are your pixel coordinate this 
        # is how to get the coordinate in space.
        #posX = px_w * x + rot1 * y + xoffset
        #posY = rot2 * x + px_h * y + yoffset
        posX = a * x + b * y + xoff
        posY = d * x + e * y + yoff
        print("",(x,y)," to ",(posX, posY))
        transformed_shape.append((posX, posY))
        # shift to the center of the pixel
        #posX += px_w / 2.0
        #posY += px_h / 2.0
        
        # get CRS from dataset 
        crs = osr.SpatialReference()
        crs.ImportFromWkt(ds.GetProjectionRef())
        # create lat/long crs with WGS84 datum
        crsGeo = osr.SpatialReference()

        crsGeo.ImportFromEPSG(3857) # pseudo marcator
        t = osr.CoordinateTransformation(crs, crsGeo)
        (lat, long, z) = t.TransformPoint(posX, posY)
        geotransformed_shape.append((lat, long))
        print("",(posX, posY)," to ", (lat, long))
        
        inv_t = osr.CoordinateTransformation(crsGeo, crs)
        inv = inv_t.TransformPoint(lat, long)
        print("",(lat, long)," to ", (round(inv[0], 1), round(inv[1], 1)))
        print()

    name = os.path.basename(img_path)
    render_path = os.path.join(render_ds_path, name)
    gdal_cmd = "gdal_translate -co BIGTIFF=YES -projwin {} {} {} {} -tr 0.333 0.333 osm_info.xml {}".format(geotransformed_shape[0][0], geotransformed_shape[0][1], geotransformed_shape[2][0], geotransformed_shape[2][1], render_path)
    source_render_path = render_path
    render_path = render_path.replace(".tif", ".png")
    gdal_convert_cmd = "gdal_translate -of png {} {}".format(source_render_path, render_path)

    name = name.replace(".tif", "")
    try:
        os.system(gdal_cmd)
        os.system(gdal_convert_cmd)
    except:
        miss_images.append(name)
    
    obtained = glob(os.path.join(render_ds_path, "*.png"))
    if not any([name in os.path.basename(o) for o in obtained]):
        miss_images.append(name)
    
with open(os.path.join(render_ds_path, 'miss_images.txt'), 'w') as f:
    for item in miss_images:
        f.write("%s\n" % item)

print("Miss Images")
print(miss_images)
print(len(miss_images))