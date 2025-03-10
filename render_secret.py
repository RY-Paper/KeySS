#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, SceneLoad
import os
import sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_hide
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import random
import numpy as np
from submodules.decoder import decoder, decoder_all, decoder_fc

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def render_set(savepath, views, gaussians, pipeline, background, dec,device):
    render_path = os.path.join(savepath, "renders")
    gts_path = os.path.join(savepath, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render_hide(dec, view, gaussians, pipeline, background)
        # image_secret, image, viewspace_point_tensor, visibility_filter, radii, visibility_filter_secret, viewspace_point_tensor_secret, radii_secret = (
        #     render_pkg["render_secret"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"],
        #     render_pkg["radii"],  render_pkg["visibility_filter_secret"], render_pkg["viewspace_points_secret"],render_pkg["radii_secret"])
        rendering = render_pkg["render_secret"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                skip_train : bool, skip_test : bool, load_path, savepath, decpth, device):
    # l = "opacity,rotation"
    # fea_all = l.split(',')
    # fealist = []
    # for f in fea_all:
    #     if f in load_path[19:]:
    #         fealist.append(f)
            
    fealist = ["opacity","rotation"] #for mipnerf
            
    dec = decoder_all(fealist) #test all kinds of features
    dec._initialize_weights()
    # dec.to(device)
    dec = torch.nn.DataParallel(dec.to("cuda"), device_ids=[0] ,output_device=0)
    dec.load_state_dict(torch.load(decpth))
    
    dataset.data_device = device 
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, device)
        scene = SceneLoad(dataset, gaussians, device,
                      load_path=load_path, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        if not skip_train:
             render_set(savepath, scene.getTrainCameras(), gaussians, pipeline, background, dec,device)

        if not skip_test:
             render_set(savepath, scene.getTestCameras(), gaussians, pipeline, background, dec,device)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--loadpath",  type=str, default=None,help="path to load ply")
    parser.add_argument("--savepath",  type=str, default=None,help="path to save render result")
    parser.add_argument("--decpth",  type=str, default=None,help="path to load dec weights")
    args = get_combined_args(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(42)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                args.skip_train, args.skip_test, args.loadpath, args.savepath, args.decpth,device)