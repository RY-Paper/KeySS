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
from gaussian_renderer import render_hide_prompt
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import random
import numpy as np
from submodules.decoder import decoder_all_prompt2 as decoder_all
import torch.nn.functional as F

def generate_prompt(text_prompt):
    # Example text prompt
    # text_prompt = "A beautiful sunset over a mountain range"
    from transformers import CLIPTextModel, CLIPTokenizer

    # Load the CLIP text encoder and tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # Tokenize the text
    inputs = tokenizer(text_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer.model_max_length)

    # Generate the text embedding
    with torch.no_grad():
        text_embeddings = text_encoder(**inputs).last_hidden_state # Example: (1, 77, 768)
    
    pooled_embeddings = text_embeddings.mean(dim=1) #1,768
    pooled_embeddings = F.normalize(pooled_embeddings.unsqueeze(-1), dim=1, eps=1e-16)
    del tokenizer, text_encoder, inputs, text_embeddings
    return pooled_embeddings.clone().detach()

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def render_set(prompt, savepath, views, gaussians, pipeline, background, dec,device):
    render_path = os.path.join(savepath, "renders")
    gts_path = os.path.join(savepath, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render_hide_prompt(prompt, dec, view, gaussians, pipeline, background)
        rendering = render_pkg["render_secret"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(prompt, dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                skip_train : bool, skip_test : bool, load_path, savepath, decpth, device, fealist):
    fealist = fealist.split(',')
            
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
             render_set(prompt, savepath, scene.getTrainCameras(), gaussians, pipeline, background, dec,device)

        if not skip_test:
             render_set(prompt, savepath, scene.getTestCameras(), gaussians, pipeline, background, dec,device)

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
    parser.add_argument("--prompt",  type=str, default=None,help="I am the key to the secret")
    parser.add_argument("--fealist",  type=str, default=None,help="scale,opacity,rotation,xyz")
    args = get_combined_args(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(42)
    print("Rendering " + args.model_path)
    
    print("------secret2 prompt:", args.prompt)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    prompt = generate_prompt(args.prompt)
    # prompt = generate_prompt("you are my sunshine")
    prompt = prompt.to(device)

    render_sets(prompt, model.extract(args), args.iteration, pipeline.extract(args), 
                args.skip_train, args.skip_test, args.loadpath, args.savepath, args.decpth,device, args.fealist)