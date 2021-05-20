import pandas as pd
from glob import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from skimage.color import label2rgb
import numpy as np
import cv2

class VisGlobalVector:
    def __init__(self, cat_path, global_vec_path):
        self.cat_dict = pd.read_csv(cat_path)
        self.global_vec_path = global_vec_path

        self.N = len(self.cat_dict.code.unique())# Number of labels
        # define the colormap
        self.cmap = plt.cm.jet
        # extract all colors from the .jet map
        self.cmaplist = [self.cmap(i) for i in range(self.cmap.N)]
        # create the new map
        self.cmap = self.cmap.from_list('Custom cmap', self.cmaplist, self.cmap.N)

        # define the bins and normalize
        self.bounds = np.linspace(0,self.N, self.N)[:self.N]
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)


    def plot_map(self, name, ax):
        img = np.load(os.path.join(self.global_vec_path,"patch_" + name + ".npy"))
        fig = ax.imshow(img -1 ,cmap=self.cmap, norm=self.norm)
        # create the colorbar
        cb = plt.colorbar(fig, ax=ax, spacing='proportional',ticks=self.bounds)
        cb.ax.set_yticklabels(self.cat_dict.category.values, fontdict={'fontsize': 25})
        ax.set_title(name, fontsize=40)

    def plot_distribution(self, name, ax):
        v = np.load(os.path.join(self.global_vec_path, name + ".npy"))
        v_df = pd.DataFrame([v], columns=list(self.cat_dict.category.values))
        v_df.T.plot.bar(ax=ax, fontsize=40, ylim=(0,1))