#! /bin/bash
. $(dirname "$(realpath $0)")/../../config_paths/paths.config
docker run -it --rm -v $ROOT_PREP:/ds_preparation \
                    -v $INRIA_ORIG:/datasets/INRIA_ORIG \
                    -v $IMAGES_PATH:/datasets/INRIA/dataset/train/images \
                    -v $GT_PATH:/datasets/INRIA/dataset/train/gt \
                    -v $VAL_IMAGES_PATH:/datasets/INRIA/dataset/val/images \
                    -v $SEMANTIC_VECTOR:/datasets/INRIA/global_descriptor_vec \
                    -v $VAL_GT_PATH:/datasets/INRIA/dataset/val/gt \
                    gdal_env /bin/bash