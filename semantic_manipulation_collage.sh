#! /bin/bash

#Generate collage

#Semantic
python ./predict_and_visualize/semantic_modification_collage.py --name semantic --vis_dir semantic_collage \
                                                                --dataset_mode satellite --label_dir /datasets/INRIA/dataset/train/gt \
                                                                --image_dir /datasets/INRIA/dataset/train/images/ \
                                                                --global_descriptor_dir /datasets/INRIA/global_descriptor_vec --label_nc 2 --gpu_ids 0 --batchSize 1 \
                                                                --contain_dontcare_label --no_instance --preprocess_mode none --no_initial_structure \
                                                                --satellite_generator_mode global_area_vector
