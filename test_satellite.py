"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_satellite_model import Pix2PixSatelliteModel
from util.visualizer import Visualizer
from util import html
from trainers.pix2pix_satellite_trainer import ClassDist
import numpy as np
import cv2
def save_mask(mask, path, mask_path):
    mask = mask.detach().cpu().numpy()[0]
    name = os.path.basename(path[0])
    path = os.path.join(mask_path, "{}.png".format(name))
    cv2.imwrite(path, mask*255)

opt = TestOptions().parse()

#class_dist = ClassDist(opt)

dataloader = data.create_dataloader(opt)

model = Pix2PixSatelliteModel(opt)

model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

#masks_path = "./masks_{}".format(opt.name)
#os.makedirs(masks_path, exist_ok=True)
# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    #class_dist.get_class_dist(data_i)
    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        #save_mask(data_i['label'][b], img_path[b:b + 1], masks_path)
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
