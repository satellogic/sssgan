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

    parser = argparse.ArgumentParser()

    parser.add_argument("--r", "--ref", type=str, default="/datasets/INRIA/dataset")

   
    args = parser.parse_args()

    path = os.path.join(args.ref, "train")
    remove_kitsap_all(path)
    path = os.path.join(args.ref, "val")
    remove_kitsap_all(path)