#!/usr/bin/env bash
## Copyright (C) 2023, KeySS
# KeySS research group, https://github.com/RY-Paper/KeySS
# All rights reserved.
#------------------------------------------------------------------------
############## single secret hiding ###############
#loadpath is the point cloud of the cover
#savepath is the path to save the render result
#decpth is the path to load the decoder
#fealist can be selected from (opacity,rotation,xyz,scale,sh)
cover_list=('bonsai')
secret_list=('counter')
modelpath=single_secret
for i in "${!cover_list[@]}"; do
    cover_item="${cover_list[$i]}"
    secret_item="${secret_list[$i]}"
    CUDA_VISIBLE_DEVICES=7 python render_cover.py -s ./data/mipnerf360/$cover_item \
    --eval \
    --skip_train \
    -m ./data/mipnerf360/$cover_item \
    --loadpath output/$modelpath/$cover_item/point_cloud_iter30000.ply \
    --savepath output/$modelpath/$cover_item/render/ 

    CUDA_VISIBLE_DEVICES=7 python render_secret_prompt_random16.py -s ./data/mipnerf360/$secret_item \
    --eval \
    --skip_train \
    --fealist sh,opacity,rotation,xyz,scale \
    --prompt "NobodysGonnaKnow" \
    -m ./data/mipnerf360/$secret_item \
    --loadpath output/$modelpath/$cover_item/point_cloud_iter30000.ply \
    --savepath output/$modelpath/$cover_item/render_secret/ \
    --decpth output/$modelpath/$cover_item/decoder_iter30000.pth 
done
CUDA_VISIBLE_DEVICES=7 python metrics_ours.py -m output/$modelpath 
CUDA_VISIBLE_DEVICES=7 python metrics_ours_secret.py -m output/$modelpath 



############## multi secret hiding ###############
#loadpath is the point cloud of the cover
#savepath is the path to save the render result
#decpth is the path to load the decoder
#fealist can be selected from (opacity,rotation,xyz,scale,sh)
cover_list=('bicycle')
secret_list=('bonsai')
secret2_list=('room')
modelpath=multi_secret
for i in "${!cover_list[@]}"; do
    cover_item="${cover_list[$i]}"
    secret_item="${secret_list[$i]}"
    secret2_item="${secret2_list[$i]}"
    CUDA_VISIBLE_DEVICES=0 python render_cover.py -s ./data/mipnerf360/$cover_item \
    --eval \
    --skip_train \
    -m ./data/mipnerf360/$cover_item \
    --loadpath output/$modelpath/$cover_item/point_cloud_iter30000.ply \
    --savepath output/$modelpath/$cover_item/render/ 

    CUDA_VISIBLE_DEVICES=0 python render_secret_prompt_random16.py -s ./data/mipnerf360/$secret_item \
    --eval \
    --skip_train \
    --fealist opacity,rotation \
    --prompt "NobodysGonnaKnow" \
    -m ./data/mipnerf360/$secret_item \
    --loadpath output/$modelpath/$cover_item/point_cloud_iter30000.ply \
    --savepath output/$modelpath/$cover_item/render_secret/ \
    --decpth output/$modelpath/$cover_item/decoder_iter30000.pth 

    CUDA_VISIBLE_DEVICES=0 python render_secret_prompt_secret2.py -s ./data/mipnerf360/$secret2_item \
    --eval \
    --skip_train \
    --prompt "HowWouldTheyFind" \
    -m ./data/mipnerf360/$secret2_item \
    --fealist opacity,rotation \
    --loadpath output/$modelpath/$cover_item/point_cloud_iter30000.ply \
    --savepath output/$modelpath/$cover_item/render_secret2/ \
    --decpth output/$modelpath/$cover_item/decoder_iter30000.pth 
done
CUDA_VISIBLE_DEVICES=0 python metrics_ours.py -m output/$modelpath  
CUDA_VISIBLE_DEVICES=0 python metrics_ours_secret.py -m output/$modelpath  
CUDA_VISIBLE_DEVICES=0 python metrics_ours_secret2.py -m output/$modelpath  

###################### 3D-Sinkhorn ####################
#evaluate the 3d sinkhorn of the cover
cover_item=bicycle
modelpath=single_secret
# Run the Python script with arguments
CUDA_VISIBLE_DEVICES=0 python render_3Dsinkhorn.py \
    -s "./data/mipnerf360/" \
    --eval \
    --skip_train \
    -m "./data/mipnerf360/" \
