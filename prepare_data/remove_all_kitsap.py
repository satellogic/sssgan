import shutil
from render_osm_classes import kitsap
from glob import glob
import os
from tqdm import tqdm

def remove_kitsap_all(path):
    #Remove

    remove_list = glob(os.path.join(path,"images", "kitsap*"))
    for r in remove_list:
        os.remove(r)
    remove_list = glob(os.path.join(path,"gt", "kitsap*"))
    for r in remove_list:
        os.remove(r)


if __name__=="__main__":
    path = "/home/emilioty/ws/datasets/INRIA/INRIA_256_128_background_filtered_without_kitsap/train/"
    remove_kitsap_all(path)
    path = "/home/emilioty/ws/datasets/INRIA/INRIA_256_128_background_filtered_without_kitsap/val/"
    remove_kitsap_all(path)