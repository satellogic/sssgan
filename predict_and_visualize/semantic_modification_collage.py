import os
from collections import OrderedDict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data
from options.test_options import TestOptions
from models.pix2pix_satellite_model import Pix2PixSatelliteModel
from util.visualizer import Visualizer
from util import html
from trainers.pix2pix_satellite_trainer import ClassDist
import pandas as pd
import matplotlib.pyplot as plt
from util.util import tensor2im
import numpy as np
import torch
from shutil import rmtree

def get_area(vec, df_area):
    idx_list = np.where(vec==1)[0]
    return df_area[df_area.idx == idx_list[0]].area.values[0]

if __name__ == "__main__":
    opt = TestOptions().parse()

    #class_dist = ClassDist(opt)

    dataloader = data.create_dataloader(opt, shuffle=True)

    model = Pix2PixSatelliteModel(opt)

    model.eval()

    N = 100
    prop = 0.8
    df_cat = pd.read_csv("/datasets/INRIA/global_descriptor_vec/metadata/vector_cat.csv")
    df_area = pd.read_csv("/datasets/INRIA/global_descriptor_vec/metadata/area_vec_idx.csv")
    df_stats = pd.read_csv("/datasets/INRIA/global_descriptor_vec/metadata/vector_analysis.csv")
    df_stats = df_stats[df_stats.columns[2:-1]]
    style_dim = len(df_cat)
    cat_to_increase = ["grass", "forest", "industrial", "rail"]

    path = os.path.join("vis", opt.vis_dir) #"./interpolation_global#area#detector_input"
    if os.path.exists(path):
        rmtree(path)

    os.makedirs(path, exist_ok=True)
    # test
    for i, data_i in enumerate(dataloader):
        if i > N: break

        #define increments
        idx_to_increase = []
        increased_vec = []
        for c in cat_to_increase:
            idx_to_increase.append(df_cat[df_cat.category == c].idx.values[0])
            quantile = df_stats[c].quantile(0.9)
            increased_vec.append(df_stats[(df_stats[c] > quantile) & (df_stats[c] < 0.998)].sample().values)


        #class_dist.get_class_dist(data_i)
        vec = data_i["global_vec"][0].cpu().numpy()
        with_area = len(vec) == style_dim + len(df_area)
        rows = len(df_area) if with_area else 1
        if with_area:
            original_area = get_area(vec[style_dim:], df_area)

        plot_grid =(rows, len(cat_to_increase)+1)
        fig, axs = plt.subplots(*plot_grid,  figsize=(50,50))

        areas = list(df_area.area.values)


        for r in range(0, rows):
            mod_data_i = data_i.copy()
            if with_area:
                area_name = areas.pop(0)
                idx_area = df_area[df_area.area == area_name].idx.values[0]
                area_vec = np.zeros(len(df_area))
                area_vec[idx_area] = 1.0
                mod_data_i["global_vec"][0][style_dim:] = torch.Tensor(area_vec)
                axs[r, 0].set_title("Area: {} //  Original Area: {}".format(area_name.upper(), original_area.upper()),fontsize=30)
            else:
                axs[r, 0].set_title("Original", fontsize=30)

            generated = model(mod_data_i, mode='inference')
            #v_df = pd.DataFrame([vec[:style_dim]], columns=list(df_cat.category.values))
            #v_df.T.plot.bar(ax=axs[0, 0], ylim=(0, 1.0), fontsize=30)
            
            
            axs[r, 0].imshow(tensor2im(generated)[0])

            
            for pos, (idx, cat, inc_vec) in enumerate(zip(idx_to_increase, cat_to_increase, increased_vec)):
                mod_data_i["global_vec"][0][:style_dim] = torch.Tensor(inc_vec)
                generated = model(mod_data_i, mode='inference')
                #v_df = pd.DataFrame([mod_data_i["global_vec"][0].cpu().numpy()[:style_dim]], columns=list(df_cat.category.values))
                #v_df.T.plot.bar(ax=axs[0, pos+1], ylim=(0, 1.0))
                axs[r, pos+1].set_title("increment {}".format(cat.upper(), prop), fontsize=30)
                axs[r, pos+1].imshow(tensor2im(generated)[0])

        i_name = os.path.basename(data_i['path'][0])
        print(i_name)
        plt.savefig(os.path.join(path, i_name))
