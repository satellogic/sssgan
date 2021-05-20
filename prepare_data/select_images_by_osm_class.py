import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import pandas as pd
import random 
from collections import defaultdict
from tqdm import tqdm
import numpy as np

path = "/home/emilioty/ws/datasets/INRIA/INRIA_256_128_background_filtered/train/global_descriptor_vectors"
metadata_path = os.path.join(path, "metadata")
sub_class_path = os.path.join(path, "metadata")
df =pd.read_csv(os.path.join(metadata_path, "osm_parsed_subclasses.csv"))
interested_in = ["retail", "commercial"]
rows = defaultdict(list)

for i in tqdm(glob(os.path.join(sub_class_path, "*.npy"))):
    name = os.path.basename(i).replace(".npy", "")
    map_im = np.load(i)
    for c in interested_in:
        code = df[df.sub_category == c].code.values[0]
        if ((map_im == code)*1).sum() > 1:
            rows[c].append([name])

for k, v in rows.items():
    d = pd.DataFrame(v, columns=["name"])
    d.to_csv("examples_with_{}.csv".format(k))
