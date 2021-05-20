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

        self.N = len(self.cat_dict.code.unique()) + 1 # Number of labels + for 0 (no code)
        # define the colormap
        self.cmap = plt.cm.jet
        # extract all colors from the .jet map
        self.cmaplist = [self.cmap(i) for i in range(self.cmap.N)]
        # create the new map
        self.cmap = self.cmap.from_list('Custom cmap', self.cmaplist, self.cmap.N)

        # define the bins and normalize
        self.bounds = np.linspace(0,self.N, self.N)[:self.N]
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)


    def plot_map(self, name, ax = None):
        img = np.load(os.path.join(self.global_vec_path,"patch_" + name + ".npy"))
        if ax is not None:
            fig = ax.imshow(img,cmap=self.cmap, norm=self.norm)
            # create the colorbar
            cb = plt.colorbar(fig, ax=ax, spacing='proportional',ticks=self.bounds)
            cb.ax.set_yticklabels(["NO_OBJ"] + list(self.cat_dict.category.values))
            ax.set_title(name)
        else:
            fig = plt.imshow(img ,cmap=self.cmap, norm=self.norm)
            # create the colorbar
            cb = plt.colorbar(fig, spacing='proportional',ticks=self.bounds)
            cb.ax.set_yticklabels(["NO_OBJ"] + list(self.cat_dict.category.values))
            plt.title(name)
        
