#! /bin/bash

#Experiments

#baseline
python train.py --name baseline --dataset_mode custom  \
                --label_dir /datasets/INRIA/datasets/train/gt \
                --image_dir /datasets/INRIA/datasets/train/images/ \
                --label_nc 2 --tf_log --gpu_ids 0,1,2,3,4,5,6,8,9 --batchSize 27 --contain_dontcare_label --no_instance  \
                --display_freq 1000 --print_freq 1000 --load_size 256 --crop_size 256 --niter 50 --save_epoch_freq 10 --no_vgg_loss \
                --cache_filelist_write --cache_filelist_read

#(SSSGAN) semantic_vector
python train_satellite.py --name semantic --dataset_mode satellite \
                          --label_dir /datasets/INRIA/datasets/train/gt \
                          --image_dir /datasets/INRIA/datasets/train/images/ \
                          --global_descriptor_dir /datasets/INRIA/global_descriptor_vec --label_nc 2 --tf_log \
                          --gpu_ids 0,1,2,3,4,5,6,8,9 --batchSize 36 --contain_dontcare_label --no_instance  --display_freq 10000 \
                          --print_freq 1000 --preprocess_mode none --niter 50 --save_epoch_freq 3  --no_vgg_loss --no_initial_structure \
                          --satellite_generator_mode global_area_vector --cache_filelist_read

#(SSSGAN) semantic_vector + dense
python train_satellite.py --name semantic#dense --dataset_mode satellite \
                          --label_dir /datasets/INRIA/datasets/train/gt \
                          --image_dir /datasets/INRIA/datasets/train/images/ \
                          --global_descriptor_dir /datasets/INRIA/global_descriptor_vec --label_nc 2 --tf_log \
                          --gpu_ids 0,1,2,3,4,5,6,8,9 --batchSize 36 --contain_dontcare_label --no_instance  --display_freq 10000 \
                          --print_freq 1000 --preprocess_mode none --niter 50 --save_epoch_freq 10 --no_vgg_loss --no_initial_structure \
                          --satellite_generator_mode global_area_vector --residual_mode --cache_filelist_write --cache_filelist_read
