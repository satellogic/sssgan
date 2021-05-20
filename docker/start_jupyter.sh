#! /bin/bash
. $(dirname "$(realpath $0)")/../config_paths/paths.config
docker run  --rm --ipc=host --gpus all -it -v $ROOT_CODE:/SSSGAN \
            -v $IMAGES_PATH:/datasets/INRIA/dataset/train/images \
            -v $GT_PATH:/datasets/INRIA/dataset/train/gt \
            -v $VAL_IMAGES_PATH:/datasets/INRIA/dataset/val/images \
            -v $VAL_GT_PATH:/datasets/INRIA/dataset/val/gt \
            -v $SEMANTIC_VECTOR:/datasets/INRIA/global_descriptor_vec \
            -p 8891:8888 \
            sssgan_env \
            jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
