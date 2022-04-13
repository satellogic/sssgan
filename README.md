<h1 align="center">SSSGAN:Satellite Style and StructureGenerative Adversarial Networks</h1>

This repository contains an end-to-end pipeline for training and using SSSGAN:Satellite Style and StructureGenerative Adversarial Networks presented in the Master Thesis [link](https://drive.google.com/drive/folders/1Qh9Ulr5kBOluwyTbaCwJP0bg_ddaTt9u?usp=sharing)(also included in folder `TFM_document` of this repository). The network was trained using [INRIA building dataset](https://project.inria.fr/aerialimagelabeling/) and [OSM](https://openstreetmap.org). SSSGAN is a deep learning GAN netowrk specially designed for generating synthetic satellite imagery. The network leverages semantic imformation specifically procesed from [OSM](https://openstreetmap.org) in semantic global vectors. This vecotors provides prior knowledge that helps the network to distangle the style latent space, helping to generate more region specific imagery. More information of the network can be find in [link](). This repository contains docker environments where all the code is executed. Morover, this reposotry contains all the training pipelines, networks, pre-porcessing and post-procesing piplines, besides the code that generates the global semantic vectors. The code is extendended from [SPADE](https://github.com/NVlabs/SPADE).

<p align="center">
<img src="./doc_img/collage_1.png" width="350" alt="CRESI">
</p>

The code is divided into two:

 * SSSGAN: explains how to train and make inferences with the network. It includes sample dataset (images , masks and semantic global vecotrs) and pre-trained weights to make inferences.

 * Semantic global vector geneartion: This section explains how to generate semantic global vecotrs

 ____

 <h1 align="center">SSSGAN</h1>

### Install ###

1. Download this repository

2. Download models and weights from [link](https://drive.google.com/drive/folders/1yh3Pv-5hvB3Jj4gTAvUEAYw9d_GsoxYq?usp=sharing)

3. Locate the folder checkoiint in the root directory and place data in any place of your local machine

4. Edit file in config_paths/paths.config:

        ROOT_CODE=/home-local/etylson/ws/sssgan_public/sssgan # Path of the root code folder
        ROOT_PREP=/home-local/etylson/ws/sssgan_public/sssgan/generate_ds # Path to the generate_ds subfolder 
        INRIA_ORIG=/data-local/data1-hdd/etylson/INRIA_original # Path to the original INRIA dataset 
        GT_PATH=/home-local/etylson/ws/sssgan_demo_data/train/gt # Path to the ground truth folder
        IMAGES_PATH=/home-local/etylson/ws/sssgan_demo_data/train/images # Path to the images  folder
        VAL_GT_PATH=/home-local/etylson/ws/sssgan_demo_data/val/gt
        VAL_IMAGES_PATH=/home-local/etylson/ws/sssgan_demo_data/val/images
        SEMANTIC_VECTOR=/home-local/etylson/ws/sssgan_demo_data/global_descriptor_vec # Path to the semantic global vector folder

5. Build docker image

		sh docker/build_environment.sh 
	
6. Run docker environment container (all commands should be run in this container)

		sh docker/start_environment.sh

### Train ###

* Training script description
  * SSGAN is trained with `train_satellite.py` script. Use `--dataset_mode` with the `satellite` option that uses ouer custo `DataSet` for satellite imagery. Specify semantic global vector location with `--label_dir`. Disbale any pre-process since the images are already preprocessed, `--preprocess_mode none`. Original spade gives as input the segmentation map, so it is needed to disable it with `--no_initial_structure`. As we are using only binary segmenataion maps use `--label_nc 2` and `--contain_dontcare_label` (we are not using ignore index) `--no_instance` (we dont use instance map). Using `no_vgg_loss` is optional, if not specified perceptual loss is used. The option `--satellite_generator_mode` controls the architecture of the network. Options for this parameter are `global_area_vector` for feeding the network the complete global semantic vector (area one hot-encoding vector and semantic classes information), `area_vector_only` to feed only area one-fot encoding information and `global_vector` to feed only semantic vector (without area information). More training parameters please refer to option files in `options`. Finally `--residual_mode` enable residual connections of the netwrok.

* Training information are stored ina folder with name spacified in the `--name` attribute under folder `./checkpoints`

* Train demo

		sh train_demo.sh

* Train full ablation study

        sh train_ablation.sh

### Test & Visualize ###

* Test script description
  * SSGAN is trained with `test_satellite.py` script. Use `--dataset_mode` with the `satellite` option that uses ouer custo `DataSet` for satellite imagery. Specify semantic global vector location with `--label_dir`. Disbale any pre-process since the images are already preprocessed, `--preprocess_mode none`. Original spade gives as input the segmentation map, so it is needed to disable it with `--no_initial_structure`. As we are using only binary segmenataion maps use `--label_nc 2` and `--contain_dontcare_label` (we are not using ignore index) `--no_instance` (we dont use instance map). Using `no_vgg_loss` is optional, if not specified perceptual loss is used. The option `--satellite_generator_mode` controls the architecture of the network. Options for this parameter are `global_area_vector` for feeding the network the complete global semantic vector (area one hot-encoding vector and semantic classes information), `area_vector_only` to feed only area one-fot encoding information and `global_vector` to feed only semantic vector (without area information). More training parameters please refer to option files in `options`.

* Visualize comparison between networks

    1. Edit configuration file for comparison `./predict_and_visualize/vis_config/vis_config.json`

        
            {
                "vis_name": "baseline", # Visualization title
                "model_name": "baseline", # name of the model
                "version": "latest", # version of the model weights
                "script": "python test.py --name {} --dataset_mode custom --label_dir {} --image_dir {} --label_nc 2  --contain_dontcare_label --no_instance --gpu_ids 2 --batchSize 1", #Scripts to be used, use {} for the script ot later fill that parameters
                "position": [1, 1] # Row q column 1 of the visualization
            },
            {
                "vis_name": "semantic", 
                "model_name": "semantic", 
                "version": "latest",
                "script": "python test_satellite.py --name {} --dataset_mode satellite --label_dir {} --image_dir {} --global_descriptor_dir /datasets/INRIA/global_descriptor_vec --label_nc 2  --contain_dontcare_label --no_instance --gpu_ids 2 --batchSize 1 --preprocess_mode none --no_initial_structure --satellite_generator_mode global_area_vector",
                "position": [1, 2]
            },

            {
                "vis_name": "semantic+dense", 
                "model_name": "semantic#dense", 
                "version": "latest",
                "script": "python test_satellite.py --name {} --dataset_mode satellite --label_dir {} --image_dir {} --global_descriptor_dir /datasets/INRIA/global_descriptor_vec --label_nc 2  --contain_dontcare_label --no_instance --gpu_ids 2 --batchSize 1 --preprocess_mode none --no_initial_structure --satellite_generator_mode global_area_vector --residual_mode",
                "position": [1, 3]
            }
    3. Run
        python ./predict_and_visualize/predict_and_visualize.py
    
    4. Results are stored in `./vis` folder

        

* Visuallize collage of semantic vector modifications

        sh semantic_manipulation_collage.sh


____

 <h1 align="center">Semantic global vector generation</h1>

### Install ###

1. Download this repository

2. Download original [INRIA building dataset](https://project.inria.fr/aerialimagelabeling/)

3. Edit file in `config_paths/paths.config`:

        ROOT_CODE=/home-local/etylson/ws/sssgan_public/sssgan # Path of the root code folder
        ROOT_PREP=/home-local/etylson/ws/sssgan_public/sssgan/generate_ds # Path to the generate_ds subfolder 
        INRIA_ORIG=/data-local/data1-hdd/etylson/INRIA_original # Path to the original INRIA dataset 
        GT_PATH=/home-local/etylson/ws/sssgan_demo_data/train/gt # Path to the ground truth folder
        IMAGES_PATH=/home-local/etylson/ws/sssgan_demo_data/train/images # Path to the images  folder
        VAL_GT_PATH=/home-local/etylson/ws/sssgan_demo_data/val/gt
        VAL_IMAGES_PATH=/home-local/etylson/ws/sssgan_demo_data/val/images
        SEMANTIC_VECTOR=/home-local/etylson/ws/sssgan_demo_data/global_descriptor_vec # Path to the semantic global vector folder

4. Build image

        sh ./generate_ds/docker/build_environment.sh

5. Run docke encironment
   
        sh ./generate_ds/docker/run_environment_gdal.sh

### Prepare dataset ###

1. Create patches of size 256 pixels and 128 stride
   
        python generate_patches.py --o /datasets/INRIA_ORIG \
                           --d /datasets/INRIA/dataset \
                           --w 256 \
                           --s 128 \

2. Download renders from OSM using images geo-location

        python download_render.py --o /datasets/INRIA_ORIG

3. Generate global semantic vectors vectors

        python generate_vectors.py --o ./osm_renders_0.3 \
                           --r /datasets/INRIA/dataset \
                           --d /datasets/INRIA/global_descriptor_vec \
                           --w 256 \
                           --s 128 
4. Remove Kitsap images

        python remove_all_kitsp.py --ref /datasets/INRIA/dataset










