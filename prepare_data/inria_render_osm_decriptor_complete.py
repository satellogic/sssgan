from render_osm_classes import class_dict, osm_render_class_color, osm_render_color_class, cat_df, sub_cat_df, subclass_to_class_dict, area_df
from glob import glob
import os
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
from PIL import Image
from shutil import rmtree
from skimage.util.shape import view_as_windows
import random

def rgb2hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def translate_image(image, class_code_dict, sub_class_code_dict, image_name):
    h, w, c = image.shape
    global_class_image = np.zeros((h, w))
    sub_class_image = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            r,g,b = image[i,j]
            hex_color = rgb2hex(r, g, b)
            sub_class = osm_render_color_class.get(hex_color, None)
            global_class = subclass_to_class_dict.get(sub_class, None)
            #Exceptions
            if "tyrol" in image_name and sub_class == "land-color":
                global_class = "grass"
            if sub_class == "park" or sub_class == "campsite":
                if random.random() < 0.4:
                    global_class = "grass"
                else:
                    global_class = "forest"

            if global_class is not None:
                global_class_image[i,j] = class_code_dict[global_class]
            if sub_class is not None:
                sub_class_image[i,j] = sub_class_code_dict[sub_class]
    return global_class_image, sub_class_image

def get_global_vector(patch, class_code_dict):
    global_vector = np.zeros(len(cat_df.code.unique()))
    h, w = patch.shape
    for global_class, global_class_code in class_code_dict.items():
        idx = global_class_code - 1
        global_vector[idx] += ((patch == global_class_code) * 1).sum()

    total = global_vector.sum() + 1e-6
    global_vector = global_vector/total
    acum_vec = list(global_vector)+[total/(h*w)]
    return global_vector, acum_vec

def generate_patch_descriptor_vector(renders, window, stride, path, class_code_dict, sub_class_code_dict, area_idx_dict, reference):
    acum_analysis = []
    for render in tqdm(renders):
        r_img = cv2.imread(render)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        r_img = cv2.resize(r_img, (5000, 5000), interpolation = cv2.INTER_NEAREST)
 
        h, w, _ = r_img.shape
        total = h*w
        name = os.path.basename(render).replace(".png","")
        #if name in reference:
        global_class_image, sub_class_image = translate_image(r_img, class_code_dict, sub_class_code_dict, name)
        window_shape = (window, window)
        global_class_patches = view_as_windows(global_class_image, window_shape, step=stride)
        window_shape = (window, window, 3)
        r_img_patches = view_as_windows(r_img, window_shape, step=stride)
        h, w = global_class_patches.shape[0], global_class_patches.shape[1]
        #area vecotor
        area_vec = np.zeros(len(area_idx_dict))
        for area, idx in area_idx_dict.items():
            if area in name:
                area_vec[idx] = 1
                break
        for i in np.arange(0, h):
            for j in np.arange(0, w):
                
                global_class_patch = global_class_patches[i,j]
                r_img_patch = r_img_patches[i,j][0]
                patch_h, patch_w = global_class_patch.shape
                if patch_h == window and patch_w == window:
                    patch_name = "{}_{}_{}".format(name, stride * i , stride* j)
                    
                    vector_descriptor, acum = get_global_vector(global_class_patch, class_code_dict)
                    
                    patch_name = "{}_{}_{}".format(name, stride * i , stride* j)
                    acum_analysis.append([patch_name] + acum)
                    

                    cv2.imwrite(os.path.join(path["global_vectors"], "original_"+patch_name+".png"),
                                cv2.cvtColor(r_img_patch, cv2.COLOR_RGB2BGR))
                    try:
                        np.save(os.path.join(path["global_vectors"], patch_name), vector_descriptor)
                        np.save(os.path.join(path["global_vectors"], "patch_"+patch_name), global_class_patch)
                        np.save(os.path.join(path["area_vector"], patch_name), area_vec)
                    except:
                        print("Error")
        np.save(os.path.join(path["sub_class_translation"], name), sub_class_image)
    return acum_analysis
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
        
    parser.add_argument("--o", "--orig", type=str, default="./osm_reders_first_version")
    parser.add_argument("--w", "--window", type=int, default=256)
    parser.add_argument("--s", "--stride", type=int, default=128)
    
    reference_path = "/home/emilioty/ws/datasets/INRIA/INRIA_256_128_background_filtered/train/images"
    sample = False

    cat_df["idx"] = range(len(cat_df.code.unique()))

    args = parser.parse_args()


    #Define file structure
    path = {}
    path["global_vectors"] = os.path.join("/home/emilioty/ws/datasets/INRIA/INRIA_256_128_background_filtered/train", "global_descriptor_vec")
    if os.path.exists(path["global_vectors"]): rmtree(path["global_vectors"])
    path["metadata"] = os.path.join(path["global_vectors"], "metadata")
    path["sub_class_translation"] = os.path.join(path["global_vectors"], "osm_subclass")
    path["area_vector"] = os.path.join(path["global_vectors"], "area_vector")

    for p in path.values():
        os.makedirs(p, exist_ok=True)


    #Generate
    cat_df.to_csv(os.path.join(path["metadata"], "vector_cat.csv"))
    sub_cat_df.to_csv(os.path.join(path["metadata"], "osm_parsed_subclasses.csv"))
    area_df.to_csv(os.path.join(path["metadata"], "area_vec_idx.csv"))

    class_code_dict = {c: cat_df[cat_df.category == c].code.values[0] for c in cat_df.category.unique()}
    sub_class_code_dict = {c: sub_cat_df[sub_cat_df.sub_category == c].code.values[0] for c in sub_cat_df.sub_category.unique()}

    area_idx_dict = {a: area_df[area_df.area == a].idx.values[0] for a in area_df.area.unique()}

    renders = glob(os.path.join(args.o,"*.png"))
    if sample:
        renders = [r for r in renders if "austin8" in r or "chicago8" in r or "tyrol-w8" in r]
        #renders = random.choice(renders, 2)

    reference = set([os.path.basename(i).split("_")[0] for i in glob(os.path.join(reference_path, "*.jpg"))])
    
    acum_vec = generate_patch_descriptor_vector(renders, args.w, args.s, path, class_code_dict, sub_class_code_dict, area_idx_dict, reference)
    cols = []
    for c in cat_df.code.unique():
        cols.append(cat_df[cat_df.code == c].category.values[0])
    acum_dc = pd.DataFrame(acum_vec, columns = ["name"] + cols+["computed_prop"])
    acum_dc.to_csv(os.path.join(path["metadata"], "vector_analysis.csv"))
    #generate_default_vector(df, {"forest": 0.475,"grass":0.475,"residential":0.05}, args.w, args.s, path)