import subprocess
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os
import random
from shutil import copyfile, rmtree
from skimage.util.shape import view_as_windows
from tqdm import tqdm
import argparse

DATASET_ARGUMENTS_TEMPLATE = "--label_dir /SSSGAN/{}/gt --image_dir /SSSGAN/{}/images --global_descriptor_dir /datasets/INRIA/global_descriptor_vec"

def red_img(from_path, to_path, window, stride, extension="jpg"): 
    for original in tqdm(glob(os.path.join(from_path, "*.{}".format(extension)))):
        o = cv2.imread(original)
        o =cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
        name = os.path.basename(original)
        window_shape = (window, window, 3)
        patch_o = view_as_windows(o,window_shape,stride)
        for i in range(patch_o.shape[0]):
            for j in range(patch_o.shape[1]):
                red_name = name.replace(".{}".format(extension),"_{}_{}.{}".format(i, j, extension))
                cv2.imwrite(os.path.join(to_path, red_name), patch_o[i, j, 0])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="semantic")
    parser.add_argument("--model_script", type=str, default="python test_satellite.py --name semantic --dataset_mode satellite \
                                                                                 --label_nc 2  --contain_dontcare_label --no_instance --gpu_ids 0,1 \
                                                                                 --batchSize 4 --preprocess_mode none --no_initial_structure \
                                                                                 --satellite_generator_mode global_area_vector")
    parser.add_argument("--generate_ds", type=str, default="True")
    args = parser.parse_args()
    
    model_name = args.model_name
    model_script = args.model_script
    generate_ds = args.generate_ds == "True"
    amount = 20000
    size = 256

    # Create Frechet reference dataset
    print("Select test images")
    test_ds = "frechet_ds"
    generate_ds = False
    test_imgs = os.path.join(test_ds, "images")
    test_gts = os.path.join(test_ds, "gt")

    # Generate a new reference dataset
    if generate_ds:
        if os.path.isdir(test_ds):
            rmtree(test_ds)
        if os.path.isdir("results"):
            rmtree("results")
        
        
        ds_path = "/datasets/INRIA/dataset/val"
        images_path = glob(os.path.join(ds_path, "images", "*jpg"))

        img_choices= random.sample(images_path, amount)
        os.makedirs(test_gts, exist_ok=True)
        os.makedirs(test_imgs, exist_ok=True)
        for img in img_choices:
            name = os.path.basename(img)
            copyfile(img, os.path.join(test_imgs,name))
            os.makedirs(test_gts, exist_ok=True)
            copyfile(os.path.join(ds_path,"gt",name), os.path.join(test_gts, name))

    #Generate example from the masks of reference dataset 
    print("TEST")
    os.system( model_script + DATASET_ARGUMENTS_TEMPLATE.format(test_ds, test_ds))

    #Compute Frechet
    print("FRECHET")
    os.system(f"python -m pytorch_fid --gpu 1 {test_imgs} /SSSSGAN/results/{model_name}/test_latest/images/synthesized_image")
