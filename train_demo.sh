#! /bin/bash
#Global + Area + Residual
python train_satellite.py --name demo_semantic#dense --dataset_mode satellite \
                          --label_dir /datasets/INRIA/dataset/train/gt \
                          --image_dir /datasets/INRIA/dataset/train/images/ \
                          --global_descriptor_dir /datasets/INRIA/global_descriptor_vec --label_nc 2 --tf_log \
                          --gpu_ids $1 --batchSize 1 --contain_dontcare_label --no_instance  --display_freq 10000 \
                          --print_freq 1000 --preprocess_mode none --niter 50 --save_epoch_freq 10 --no_vgg_loss --no_initial_structure \
                          --satellite_generator_mode global_area_vector --residual_mode  
