import subprocess
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os
import random
from shutil import copyfile, rmtree, move
from vis_global_vectors import VisGlobalVector
import sys
# sys.path.insert(0, "/SSSGAN/trainers")
# from class_dist_regularizer import SatelliteDistributionRegularizer
import pandas as pd
import argparse
import json

def generate_samples(subsamples, vis_path, predict=True, amount=200):
    plt_qty = len(subsamples) + 1
    plot_grid =(2, plt_qty)
    ###########################################
    ######## Create test sample images ########
    ###########################################

    print("Select test images")
    test_ds = "test_samples"
    if os.path.isdir(test_ds):
        rmtree(test_ds)
    test_imgs = os.path.join(test_ds, "images")
    test_gts = os.path.join(test_ds, "gt")
    os.makedirs(test_ds, exist_ok=True)
    os.makedirs(test_imgs, exist_ok=True)
    os.makedirs(test_gts, exist_ok=True)
    ds_path = "/datasets/INRIA/dataset/train"
    images_path = glob(os.path.join(ds_path, "images", "*jpg"))
    img_choices= random.choices(images_path, k=amount)


    for img in img_choices:
        name = os.path.basename(img)
        copyfile(img, os.path.join(test_imgs,name))
        copyfile(os.path.join(ds_path,"gt",name), os.path.join(test_gts,name))

    #########################
    ######## Predict ########
    #########################
    if os.path.exists("./results"): rmtree("./results")
    if predict:
        if os.path.isdir(vis_path): rmtree(vis_path)
        os.makedirs(vis_path, exist_ok=True)

        for (name, model_name,checkpoint_v, script, _) in subsamples:
            print("TEST")
            os.system(script.format(model_name, test_gts, test_imgs))
        
            #Move predictions
            print("VIS")
            results_path_synth = "./results/{}/test_{}/images/synthesized_image".format(model_name, checkpoint_v)

            path = os.path.join(vis_path, name)
            if os.path.isdir(path): rmtree(path)
            os.makedirs(path, exist_ok=True)
            pred_path = os.path.join(path, "predictions")
            move(results_path_synth, pred_path)
            

        #Move original test sampe ds
        original_path = os.path.join(vis_path, "original")
        move(test_ds, original_path)
    
    ###########################
    ######## Visualize ########
    ###########################

    print("Generate collage")

    original_path = os.path.join(vis_path, "original")
    original_img = os.path.join(original_path, "images")
    original_gt = os.path.join(original_path, "gt")
    global_vec_vis = VisGlobalVector("/datasets/INRIA/global_descriptor_vec/metadata/vector_cat.csv", "/datasets/INRIA/global_descriptor_vec")
    # dist_model = SatelliteDistributionRegularizer()

    for mask_name in glob(os.path.join(original_gt, "*.jpg")):
    
        mask = cv2.imread(mask_name)
        mask = mask * 255
        name = os.path.basename(mask_name)
        #dist_df_list = []
        if os.path.exists("/datasets/INRIA/global_descriptor_vec/patch_"+name.replace(".jpg","")+".npy"):
            print("Processing ", name)
            real_img = os.path.join(original_img, name)
            #dist_df_list.append(dist_model.count_classes(real_img, "real"))
            real_img = cv2.imread(real_img)
            real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)

            
            fig, axs = plt.subplots(*plot_grid,  figsize=(50,50))
            axs[0, 0].set_title("mask", fontsize=40)
            axs[0, 0].imshow(mask)

            axs[1, 0].set_title("real image", fontsize=40)
            axs[1, 0].imshow(real_img)

            global_vec_vis.plot_map(name.replace(".jpg",""), axs[0, 1])
            global_vec_vis.plot_distribution(name.replace(".jpg",""), axs[0, 3])

            fig.delaxes(axs[0][2])

            #Vis predictions
            for i, (folder_name, model_name, checkpoint_v, script, position) in enumerate(subsamples):
                i_name = name.replace("jpg","png")
                synth_img = os.path.join(vis_path, folder_name, "predictions", i_name)
                #dist_df_list.append(dist_model.count_classes(synth_img, model_name))

                synth_img = cv2.imread(synth_img)
                synth_img = cv2.cvtColor(synth_img, cv2.COLOR_RGB2BGR)

                axs[position[0], position[1]].set_title(folder_name, fontsize=40)
                axs[position[0], position[1]].imshow(synth_img)
            
            # df = pd.concat(dist_df_list)
            # df.T.plot.bar(ax=axs[2, 1])
            plt.savefig(os.path.join(vis_path, i_name))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    current_path = os.path.dirname(__file__)
    default_path = os.path.join(current_path, "vis_config", "vis_config.json")
    parser.add_argument("--conf_path", type=str, default=default_path)
    args = parser.parse_args()

    config = json.load(open(args.conf_path))
    subsamples = [ [conf["vis_name"], conf["model_name"], conf["version"],
                   conf["script"], conf["position"]] for conf in config]
    name = "#".join([ conf[0] for conf in subsamples])
    vis_path = "./vis/{}".format(name)

    generate_samples(subsamples, vis_path, predict=True, amount=100)



