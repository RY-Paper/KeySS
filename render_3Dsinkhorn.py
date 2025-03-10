#!/usr/bin/env bash
## Copyright (C) 2023, KeySS
# KeySS research group, https://github.com/RY-Paper/KeySS
# All rights reserved.
#------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

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
import matplotlib.pyplot as plt
from gaussian_renderer import render
from scipy.stats import chisquare
from scipy.linalg import sqrtm
from utils.general_utils import inverse_sigmoid
import pandas as pd
import ot
cover_list = ['bonsai']
secret_list = ['counter']


def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def render_set(prompt, random_prompt, savepath, views, gaussians, pipeline, background, dec,device):
    render_path = os.path.join(savepath, "renders")
    gts_path = os.path.join(savepath, "gt")
    # random_path = os.path.join(savepath, "random")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))




def plot_histogram(opacity_tensor, figname):

    # Convert the tensor to a NumPy array
    opacity_values = opacity_tensor.cpu().numpy().flatten()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(opacity_values, bins=100, range=(0, 1), color='blue', alpha=0.7, edgecolor='black')

    # Add labels and title
    plt.xlabel('Opacity Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Opacity Values')

    # Save the plot as a PNG file
    plt.savefig(f"{figname}.png", format='png', dpi=300)
    
def plot_histogram_all(gaussians_init_opacity, gaussians_opacity, figname):
    # Convert tensors to NumPy arrays
    init_opacity_values = gaussians_init_opacity.cpu().numpy().flatten()
    pred_opacity_values = gaussians_opacity.cpu().numpy().flatten()

    # Plot the histograms
    plt.figure(figsize=(10, 6))
    bins = 100  # Number of bins

    # Plot initial opacity histogram
    plt.hist(init_opacity_values, bins=bins, range=(0, 1), alpha=0.5, color='blue', label='Initial Opacity', density=True)

    # Plot predicted opacity histogram
    plt.hist(pred_opacity_values, bins=bins, range=(0, 1), alpha=0.5, color='red', label='Predicted Opacity', density=True)

    # Add labels and title
    plt.xlabel('Opacity Value')
    plt.ylabel('Density')
    plt.title('Histogram of Opacity Values')
    plt.legend(loc='upper right')

    # Save the plot as a PNG file
    plt.savefig(f"{figname}.png", format='png', dpi=300)

def histogram_distance(gaussians_init_opacity, gaussians_opacity, range):
    # Convert tensors to NumPy arrays
    init_opacity_values = gaussians_init_opacity.cpu().numpy().flatten()
    pred_opacity_values = gaussians_opacity.cpu().numpy().flatten()

    # Compute histograms
    bins = 100  # Number of bins
    init_hist, _ = np.histogram(init_opacity_values, bins=bins, range=range, density=True)
    pred_hist, _ = np.histogram(pred_opacity_values, bins=bins, range=range, density=True)
    init_hist_normalized = init_hist / (init_hist.sum() + 1e-10) + 1e-10
    pred_hist_normalized = pred_hist / (pred_hist.sum() + 1e-10) + 1e-10

    # Calculate Chi-Squared distance
    chi_squared_distance, _ = chisquare(init_hist_normalized, pred_hist_normalized)

    # print("Chi-Squared Distance:", chi_squared_distance)
    return chi_squared_distance

# Initialize an empty DataFrame to store results for all methods and covers
global results_df
results_df = pd.DataFrame(columns=[
    "method", "cover", "sinkhorn_distance", "opacity_sinkhorn",
    "scale_sinkhorn", "rotation_sinkhorn",
    "xyz_sinkhorn", "sh_sinkhorn"
])

def compute_sinkhorn_distance(dist1, dist2, num_bins=100, reg=0.01):
    """
    Compute Sinkhorn distance using POT's implementation with numerical stability
    """
    # Replace NaN values with 1
    dist1 = torch.nan_to_num(dist1, nan=1.0)
    dist2 = torch.nan_to_num(dist2, nan=1.0)
    
    # Compute histograms
    hist1, bin_edges = torch.histogram(dist1.flatten().cpu(), bins=num_bins, density=True)
    hist2, _ = torch.histogram(dist2.flatten().cpu(), bins=bin_edges, density=True)
    
    # Convert to probabilities with numerical stability
    p = hist1.numpy() + 1e-10  # Add small constant
    q = hist2.numpy() + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    
    # Compute cost matrix
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).numpy()
    M = np.abs(bin_centers[:, None] - bin_centers[None, :])
    
    try:
        # Compute Sinkhorn distance with numerical stability
        distance = ot.sinkhorn2(p, q, M, reg, 
                              numItermax=100,
                              method='sinkhorn',
                              stopThr=1e-6,
                              verbose=False)
    except:
        # Fallback to larger regularization if algorithm fails
        distance = ot.sinkhorn2(p, q, M, reg*10,
                              numItermax=50,
                              method='sinkhorn_stabilized',
                              stopThr=1e-6,
                              verbose=False)
    
    return float(distance)

def compute_all_distances(gaussians, gaussians_init):
    """Compute Sinkhorn distances for all Gaussian attributes"""
    #normalize to [0,1], then compute sinkhorn distance
    distances = {}
    
    # Opacity distance
    opacity = gaussians.get_opacity
    opacity_init = gaussians_init.get_opacity
    distances['opacity'] = compute_sinkhorn_distance(
        opacity,
        opacity_init
    )
    
    # Scale distance (normalized)
    scale = torch.linalg.norm(gaussians.get_scaling,dim=-1,keepdim=True)
    scale_init = torch.linalg.norm(gaussians_init.get_scaling,dim=-1,keepdim=True)
    distances['scale'] = compute_sinkhorn_distance(
        scale/scale.max(),
        scale_init/scale_init.max()
    )
    
    # Rotation distance
    rotation = torch.linalg.norm(gaussians._rotation,dim=-1,keepdim=True)
    rotation_init = torch.linalg.norm(gaussians_init._rotation,dim=-1,keepdim=True)
    distances['rotation'] = compute_sinkhorn_distance(
        rotation/rotation.max(),
        rotation_init/rotation_init.max()
    )
    
    # XYZ distance (normalized)
    xyz = torch.linalg.norm(gaussians._xyz,dim=-1,keepdim=True)
    xyz_init = torch.linalg.norm(gaussians_init._xyz,dim=-1,keepdim=True)
    distances['xyz'] = compute_sinkhorn_distance(
        xyz/xyz.max(),
        xyz_init/xyz_init.max()
    )
    
    # SH distance (normalized)
    sh_norm = torch.linalg.norm(gaussians._features_dc.view(-1,3,1).squeeze(-1),dim=-1,keepdim=True)
    sh_init_norm = torch.linalg.norm(gaussians_init._features_dc.view(-1,3,1).squeeze(-1),dim=-1,keepdim=True)
    distances['sh'] = compute_sinkhorn_distance(
        sh_norm/sh_norm.max(),
        sh_init_norm/sh_init_norm.max()
    )
    
    # Average distance
    distances['average'] = sum(distances.values()) / len(distances)
    
    return distances

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                skip_train: bool, skip_test: bool, load_path, inipath, device, method, cover):
    global results_df
    dataset.data_device = device
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, device)
        scene = SceneLoad(dataset, gaussians, device, load_path=load_path, shuffle=False)
        
        gaussians_init = GaussianModel(dataset.sh_degree, device)
        scene_init = SceneLoad(dataset, gaussians_init, device, load_path=inipath, shuffle=False)
        
        # Compute Sinkhorn distances
        distances = compute_all_distances(gaussians, gaussians_init)
        
        # Add results to DataFrame
        new_row = pd.DataFrame([{
            "method": method,
            "cover": cover,
            "sinkhorn_distance": distances['average'],
            "opacity_sinkhorn": distances['opacity'],
            "scale_sinkhorn": distances['scale'],
            "rotation_sinkhorn": distances['rotation'],
            "xyz_sinkhorn": distances['xyz'],
            "sh_sinkhorn": distances['sh']
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        
        print(f"Sinkhorn distances - Average: {distances['average']:.3f}, "
              f"Opacity: {distances['opacity']:.3f}, "
              f"Scale: {distances['scale']:.3f}, "
              f"Rotation: {distances['rotation']:.3f}, "
              f"XYZ: {distances['xyz']:.3f}, "
              f"SH: {distances['sh']:.3f}")
        
        return distances['average']

def compute_method_averages(results_df):
    # Group by method and compute mean for all numeric columns
    method_averages = results_df.groupby('method').agg({
        'sinkhorn_distance': 'mean',
        'opacity_sinkhorn': 'mean',
        'scale_sinkhorn': 'mean',
        'rotation_sinkhorn': 'mean',
        'xyz_sinkhorn': 'mean',
        'sh_sinkhorn': 'mean'
    }).round(4)  # Round to 4 decimal places
    
    # Add a column to identify these rows as averages
    method_averages['cover'] = 'AVERAGE'
    
    # Concatenate with original results
    final_results = pd.concat([results_df, method_averages.reset_index()])
    
    # # Save both detailed and summary results
    final_results.to_csv("sinkhorn_distances_by_method_and_cover_sinkhorn_check.csv", index=False)
    final_results.to_excel("sinkhorn_distances_by_method_and_cover_sinkhorn_check.xlsx", index=False, engine='openpyxl')
    
    # Print averages
    print("\nMethod Averages:")
    print(method_averages)
    
    return method_averages

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(42)
    # print("Rendering " + args.model_path)
    method_list = [
        "single_secret",
    ]
    cover_list = [
        "bicycle",
    ]
    
    # Example loop over methods and covers
    for method in method_list:
        for cover in cover_list:
            sys.argv[2] ="./data/mipnerf360/" + cover
            sys.argv[6] ="./data/mipnerf360/" + cover
            args = get_combined_args(parser)
            safe_state(args.quiet)
            # Call render_sets with method and cover
            args.loadpath = "output/" + method + "/" + cover + "/point_cloud_iter30000.ply"
            args.inipath = "output/mipnerf360/" + cover + "/point_cloud/iteration_30000/point_cloud.ply"
            args.savepath = "output/" + method + "/" + cover + "/render_histogram/"
            print("======method:", args.savepath)
            render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                        args.skip_train, args.skip_test, args.loadpath, args.inipath, 
                        device, method, cover)
    
            # Save the complete DataFrame to a CSV file
            results_df.to_csv("sinkhorn_distances_by_method_and_cover_sinkhorn_check.csv", index=False)
            results_df.to_excel("sinkhorn_distances_by_method_and_cover_sinkhorn_check.xlsx", index=False, engine='openpyxl')

    # After all methods and covers are processed
    compute_method_averages(results_df)
