"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_satellite_dataset import Pix2pixSatelliteDataset
from data.image_folder import make_dataset
import os
from glob import glob


class SatelliteDataset(Pix2pixSatelliteDataset):
    """ Dataset that loads satellital images from directories
        Use option --label_dir, --image_dir, --instance_dir, --global_descriptor to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixSatelliteDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        parser.add_argument('--global_descriptor_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        # Global Vectors
        if len(opt.global_descriptor_dir) > 0:
            global_descriptor_dir = opt.global_descriptor_dir
            global_descriptor_paths = glob(os.path.join(global_descriptor_dir,"*.npy"))
            #match global descriptor vectors to images
            names = set([os.path.basename(p).replace(".jpg", "") for p in image_paths])
            global_descriptor_paths = { os.path.basename(g).replace(".npy", ""): g for g in global_descriptor_paths if os.path.basename(g).replace(".npy", "") in names }
            if (opt.satellite_generator_mode == "global_area_vector" or opt.satellite_generator_mode == "global_area_class_vector" ) and len(global_descriptor_paths) > 0:
                area_descriptor_dir  = os.path.join(global_descriptor_dir, "area_vector")
                area_descriptor_paths = glob(os.path.join(area_descriptor_dir,"*.npy"))
                #match global descriptor vectors to images
                area_descriptor_paths = { os.path.basename(a).replace(".npy", ""): a for a in area_descriptor_paths if os.path.basename(a).replace(".npy", "") in names }
            else:
                area_descriptor_paths = {}
        else:
            global_descriptor_paths = {}



        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"
        assert len(global_descriptor_paths) >= len(image_paths), "The #global_vectors in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths, global_descriptor_paths, area_descriptor_paths
