#! /bin/bash

python download_render.py --o /datasets/INRIA_ORIG
python generate_vectors.py --o ./osm_renders_0.3 --r /datasets/INRIA/dataset --d /datasets/INRIA/global_descriptor_vec --w 256 --s 128 