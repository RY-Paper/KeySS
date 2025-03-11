#!/usr/bin/env bash
## Copyright (C) 2023, KeySS
# KeySS research group, https://github.com/RY-Paper/KeySS
# All rights reserved.
#------------------------------------------------------------------------
############## single secret hiding ###############
# all_path is the point cloud of the secret
# hide_path is the point cloud of the cover
# source_path_secret is the path of the secret initial info
#fealist can be selected from (opacity,rotation,xyz,scale,sh)
cover_list=('bonsai')
secret_list=('counter')
for j in {1}; do
    for i in "${!cover_list[@]}"; do
        cover_item="${cover_list[$i]}"
        secret_item="${secret_list[$i]}"
        CUDA_VISIBLE_DEVICES=0 python train2hide_prompt_random_fast16.py -s ./data/mipnerf360/$cover_item \
        --port 7988 --eval \
        --fealist opacity,rotation,xyz,scale \
        --prompt "NobodysGonnaKnow" \
        --save_path output/single_secret/$cover_item \
        --all_path output/mipnerf360/$secret_item/point_cloud/iteration_30000/point_cloud.ply \
        --source_path_secret data/mipnerf360/$secret_item \
        --hide_path output/mipnerf360/$cover_item/point_cloud/iteration_30000/point_cloud.ply
   done
done

for single secret hiding with sh feature, due to gpu limit, please run the following command:
cover_list=('bonsai')
secret_list=('counter')
for j in {1}; do
    for i in "${!cover_list[@]}"; do
        cover_item="${cover_list[$i]}"
        secret_item="${secret_list[$i]}"
        CUDA_VISIBLE_DEVICES=0 python train2hide_prompt_random_fast16_sh.py -s ./data/mipnerf360/$cover_item \
        --port 7988 --eval \
        --fealist sh,opacity,rotation,xyz,scale \
        --prompt "NobodysGonnaKnow" \
        --save_path output/single_secret/$cover_item \
        --all_path output/mipnerf360/$secret_item/point_cloud/iteration_30000/point_cloud.ply \
        --source_path_secret data/mipnerf360/$secret_item \
        --hide_path output/mipnerf360/$cover_item/point_cloud/iteration_30000/point_cloud.ply
    done
done



############## multi secret hiding ###############
# all_path is the point cloud of the secret
# secret2_path is the point cloud of the secret2
# hide_path is the point cloud of the cover
# source_path_secret is the path of the secret initial info
# source_path_secret2 is the path of the secret2 initial info
# fealist can be selected from (opacity,rotation,xyz,scale,sh)
cover_list=('bicycle')
secret_list=('bonsai')
secret2_list=('room')
for i in "${!cover_list[@]}"; do
    cover_item="${cover_list[$i]}"
    secret_item="${secret_list[$i]}"
    secret2_item="${secret2_list[$i]}"
    CUDA_VISIBLE_DEVICES=0 python train2hide_prompt_random_hide2_faster16.py -s ./data/mipnerf360/$cover_item \
    --port 9080 --eval \
    --fealist opacity,rotation,xyz,scale \
    --save_path output/multi_secret/$cover_item \
    --all_path output/mipnerf360/$secret_item/point_cloud/iteration_30000/point_cloud.ply \
    --secret2_path output/mipnerf360/$secret2_item/point_cloud/iteration_30000/point_cloud.ply \
    --source_path_secret data/mipnerf360/$secret_item \
    --source_path_secret2 data/mipnerf360/$secret2_item \
    --hide_path output/mipnerf360/$cover_item/point_cloud/iteration_30000/point_cloud.ply 
done

