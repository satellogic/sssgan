import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import argparse

def write_patch(img, name, i, j, output_path):
    name = name.split(".")[0]
    name = "{}_{}_{}.jpg".format(name, i, j)
    path = os.path.join(output_path, name)
    cv2.imwrite(path, img)


def patch_generator(mask_path, image_path, 
                    image_output_path, mask_output_path, window=512, stride=256, filter_all_bkg=False):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    image_name = os.path.basename(image_path)
    mask_name = os.path.basename(mask_path)

    h, w, c = image.shape

    for i in np.arange(0, h, stride):
        for j in np.arange(0, w, stride):
            patch_image = image[i:i+window, j:j+window]
            patch_mask = mask[i:i+window, j:j+window]
            patch_mask = patch_mask / 255
            patch_mask[patch_mask>0.9] = 1
            patch_mask[patch_mask<0.1] = 0
            patch_h, patch_w, c = patch_image.shape

            if patch_h == window and patch_w == window:
                if patch_mask.sum() > 0.01 and patch_mask.sum() < 0.1: print("all_bkg")
                if filter_all_bkg and patch_mask.sum() < 0.1:
                    continue
                else:
                    write_patch(patch_image, image_name, i, j, image_output_path)
                    write_patch(patch_mask, image_name, i, j, mask_output_path)


class INRIA:
    def __init__(self, orig_ds_path, output_ds_path):
        self.orig_ds_path = orig_ds_path
        self.orig_ds_path_train = os.path.join(self.orig_ds_path, "train")
        self.output_ds_path = output_ds_path
        self.output_train = os.path.join(self.output_ds_path, "train")
        self.output_val = os.path.join(self.output_ds_path, "val")

        self.output_img_train = os.path.join(self.output_train, "images")
        self.output_mask_train =os.path.join(self.output_train, "gt")

        self.output_img_val = os.path.join(self.output_val, "images")
        self.output_mask_val =os.path.join(self.output_val, "gt")

        os.makedirs(self.output_train, exist_ok = True)
        os.makedirs(self.output_val, exist_ok = True)

        os.makedirs(self.output_img_train, exist_ok = True)
        os.makedirs(self.output_mask_train, exist_ok = True)

        os.makedirs(self.output_img_val, exist_ok = True)
        os.makedirs(self.output_mask_val, exist_ok = True)

    def generate_dataset(self, window, stride, train_augmentation = None, val_augmentation= None, filter_all_bkg=False):
        ds={}
        already_contains_imgs = len(glob.glob(os.path.join(self.output_img_train, "images"))) > 0
        if not already_contains_imgs:
            imgs = glob.glob(os.path.join(self.orig_ds_path_train, "images","*.tif"))
            imgs = np.array([os.path.basename(i) for i in imgs])
            idxs = np.random.permutation(range(len(imgs)))

            txt =  imgs[idxs[:144]]
            self._generate_patches(txt, self.output_img_train, self.output_mask_train, window, stride, filter_all_bkg=filter_all_bkg)
            txt =  imgs[idxs[144:]] 
            self._generate_patches(txt, self.output_img_val, self.output_mask_val, window, stride, filter_all_bkg=filter_all_bkg)
    
    def _generate_patches(self, txt, out_img, out_mask, window, stride, filter_all_bkg=False):
        for image_name in tqdm(txt):
            mask_path = os.path.join(self.orig_ds_path_train , "gt", image_name)
            image_path = os.path.join(self.orig_ds_path_train , "images", image_name)
            patch_generator(
                mask_path=mask_path,
                image_path=image_path, 
                image_output_path=out_img,
                mask_output_path=out_mask,
                window=window,
                stride=stride,
                filter_all_bkg=filter_all_bkg
            )

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    #--o /scratch/eva/TENCENT/Inria_dataset/AerialImageDataset --d /home/emilioty/ws/datasets/INRIA/INRIA_256_128_background_filtered
    
    parser.add_argument("--o", "--origin", type=str, required=True)
    parser.add_argument("--d", "--dest", type=str, required=True)
    parser.add_argument("--w", "--window", type=int, default=256)
    parser.add_argument("--s", "--stride", type=int, default=128)


    args = parser.parse_args()

    origin = args.o
    dest = args.d

    inria = INRIA(orig_ds_path=origin, output_ds_path=dest)
    inria.generate_dataset(window=args.w, stride=args.s, filter_all_bkg=True)