# Copyright (C) 2023, KeySS
# KeySS research group, https://github.com/RY-Paper/KeySS
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
import gc
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_hide_prompt, render_imageonly, render_hide_prompt_hide2
import sys
from scene import Scene, GaussianModel, SceneLoad, SceneLoadBoth2
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json
# from edit_object_hide import hide_setup, hide_setup_all, hide_setup_inverse
from submodules.decoder import decoder_all_prompt2 as decoder_all
import torchvision
from utils.system_utils import mkdir_p
import random
import numpy as np
import torch.nn.functional as F
from submodules.prompt import generate_prompt_batch
from transformers import CLIPTextModel, CLIPTokenizer
import string

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
def generate_prompt(text_prompt):
    # Load CLIP models and set them to eval mode
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder.eval()  # Set to eval mode
    
    # Freeze all CLIP parameters
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    # Move to CUDA and process with no gradients
    text_encoder = text_encoder.cuda()
    with torch.no_grad():
        # Tokenize the text
        inputs = tokenizer(text_prompt, return_tensors="pt", padding="max_length", 
                         truncation=True, max_length=tokenizer.model_max_length).to('cuda')
        
        # Generate embeddings
        text_embeddings = text_encoder(**inputs).last_hidden_state
        pooled_embeddings = text_embeddings.mean(dim=1)
        pooled_embeddings = F.normalize(pooled_embeddings.unsqueeze(-1), dim=1, eps=1e-16)
    
    # Clean up and ensure embeddings are detached
    del tokenizer, text_encoder, inputs, text_embeddings
    torch.cuda.empty_cache()
    
    return pooled_embeddings.clone().detach()

def generate_randomprompt(tokenizer, text_encoder, batch_size=8):
    # Ensure models are in eval mode and frozen
  
    # Generate random tokens for each batch
    batch_token_ids = []
    for _ in range(batch_size):
        num_tokens = random.randint(15, 30)
        random_token_ids = np.random.randint(0, tokenizer.vocab_size, size=(num_tokens,)).tolist()
        eos_token_id = tokenizer.eos_token_id
        padded_token_ids = random_token_ids + [eos_token_id] * (77 - len(random_token_ids))
        batch_token_ids.append(padded_token_ids)
        
        # # Create batched input
        # random_text = tokenizer.decode(random_token_ids, skip_special_tokens=True)
        # print("Random Text:", random_text)
    
    
    token_ids = torch.tensor(batch_token_ids).to('cuda')
    inputs = {'input_ids': token_ids}
    
    # Generate embeddings in batch
    text_embeddings = text_encoder(**inputs).last_hidden_state
    pooled_embeddings = text_embeddings.mean(dim=1)
    pooled_embeddings = F.normalize(pooled_embeddings.unsqueeze(-1), dim=1, eps=1e-16)
    
    return pooled_embeddings.clone().detach()

def training(prompt, prompt2, random_prompts, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
             debug_from, use_wandb, save_path: str,
             allply:str, hideply:str, secret2ply:str, device, fealist, w_cover, w_secret, w_secret2, w_random,
             source_path_secret=None, source_path_secret2=None):
    first_iter = 0
    dataset.source_path_secret = source_path_secret
    dataset.source_path_secret2 = source_path_secret2
    prepare_output_and_logger(dataset)
    
    # Load the CLIP text encoder and tokenizer
    
    # Get mean scaling from secret gaussians
    gaussians_secret = GaussianModel(dataset.sh_degree, device)
    scene_secret = SceneLoad(dataset, gaussians_secret, device, load_path=allply, shuffle=False)
    mean_scaling = gaussians_secret.get_scaling.mean().item()
    gaussians_secret.freeze()
    with torch.no_grad():
        scales_secret_gt = gaussians_secret.get_scaling.clone().detach()
        # opacity_secret_gt = gaussians_secret.get_opacity.clone().detach()
    del scene_secret
    torch.cuda.empty_cache()
    
    # Get mean scaling from secret gaussians
    gaussians_secret2 = GaussianModel(dataset.sh_degree, device)
    scene_secret2 = SceneLoad(dataset, gaussians_secret2, device, load_path=secret2ply, shuffle=False)
    gaussians_secret2.freeze()
    with torch.no_grad():
        scales_secret2_gt = gaussians_secret2.get_scaling.clone().detach()
        # opacity_secret2_gt = gaussians_secret2.get_opacity.clone().detach()
    del scene_secret2
    torch.cuda.empty_cache()

    #init decoder
    dec = decoder_all(fealist, mean_scaling=mean_scaling)
    dec._initialize_weights()
    
    # Create parameter groups with different learning rates
    prompt_params = [p for n, p in dec.named_parameters() if 'prompt_embedding' in n]
    other_params = [p for n, p in dec.named_parameters() if 'prompt_embedding' not in n]
    
    param_groups = [
        {'params': prompt_params, 'lr': 5e-5},  # Slower for prompt compression
        {'params': other_params, 'lr': 5e-5}    # Faster for feature transformation
    ]
    
    # Add stronger regularization
    dec_optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    dec = torch.nn.DataParallel(dec.to("cuda"), device_ids=[0], output_device=0)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # prepare the gaussian remain
    gaussians_remain = GaussianModel(dataset.sh_degree, device)
    scene_remain = SceneLoad(dataset, gaussians_remain, device,
                         load_path=hideply, shuffle=False)
    gaussians_remain.freeze()
    del scene_remain
    torch.cuda.empty_cache()

    #gs for train
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = SceneLoadBoth2(dataset, gaussians, device) #, load_path=hideply)
    gaussians.training_setup(opt, device)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device=device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    #get gt ready
    # seed_torch(42)
    viewpoint_stack = scene.getTrainCameras()
    for viewpoint_cam in viewpoint_stack: #ori is better
        with torch.no_grad(),torch.cuda.amp.autocast():
            gt_image = render_imageonly(viewpoint_cam, gaussians_remain, pipe, background)
            gt_secret = render_imageonly(viewpoint_cam, gaussians_secret, pipe, background)
            gt_secret2 = render_imageonly(viewpoint_cam, gaussians_secret2, pipe, background)
            mse_cover = F.mse_loss(gt_image, viewpoint_cam.original_image)
            mse_secret = F.mse_loss(gt_secret, viewpoint_cam.original_image)
            # mse_secret2 = F.mse_loss(gt_secret2, viewpoint_cam.original_image)
            if mse_secret < mse_cover:
                viewpoint_cam.gt_secret = viewpoint_cam.original_image.detach().clone().cpu()
                viewpoint_cam.gt_cover = gt_image.cpu()
                viewpoint_cam.gt_secret2 = gt_secret2.cpu()
            else:
                viewpoint_cam.gt_cover = viewpoint_cam.original_image.detach().clone().cpu()
                viewpoint_cam.gt_secret = gt_secret.cpu()
                viewpoint_cam.gt_secret2 = gt_secret2.cpu()
            del viewpoint_cam.original_image
            
            torch.cuda.empty_cache()
    del gt_image, gt_secret, gt_secret2
    del gaussians_remain, gaussians_secret, gaussians_secret2, scene.train_cameras
    torch.cuda.empty_cache()
    
    sequence = []
    indices = list(range(len(viewpoint_stack)))
    while len(sequence) < opt.iterations+1:
        random.shuffle(indices)  # Randomize the order of the list
        sequence.extend(indices)
    sequence = sequence[:opt.iterations+1]
    
    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    
    for iteration in range(first_iter, opt.iterations + 1):       
        # #random prompt
        # random_prompt = random_prompts[iteration].to(device)

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        viewpoint_cam = viewpoint_stack[sequence[iteration]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        try:
           
            # with torch.cuda.amp.autocast():
            render_pkg = render_hide_prompt_hide2(
                    prompt, prompt2, random_prompts[iteration].to(device),
                    dec, viewpoint_cam, gaussians, pipe, background
                )
        
            with torch.no_grad():
                gt_image = viewpoint_cam.gt_cover.to(device)
                gt_secret = viewpoint_cam.gt_secret.to(device)
                gt_secret2 = viewpoint_cam.gt_secret2.to(device)
            
            # Loss
            # Calculate losses
            loss, loss_cover, loss_secret, loss_random, loss_secret2  = compute_losses(render_pkg, gt_image, gt_secret, gt_secret2, 
                                scales_secret_gt, scales_secret2_gt,
                                opt, w_cover, w_secret, w_secret2, w_random)
            del gt_image, gt_secret, gt_secret2     
            torch.cuda.empty_cache()
               

            loss.backward()
            iter_end.record()
            # print("loss1:", loss1, " ,loss2:" ,loss2)

            with torch.no_grad():
                # Progress bar
                # gc.collect()
                # torch.cuda.empty_cache()
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 100 == 0:
                    progress_bar.set_postfix({"Loss_cover": f"{loss_cover:.{3}f}", "Loss_secret": f"{loss_secret:.{3}f}", "Loss_secret2": f"{loss_secret2:.{3}f}", "Loss_random": f"{loss_random:.{3}f}", "gs_num":f"{gaussians._scaling.shape[0]}"})
                    progress_bar.update(100)
                if iteration == opt.iterations:
                    progress_bar.close()
                    
                if iteration % 5000==0 and iteration >= 10000:
                    scene.save(iteration, save_path) #save the .ply
                    torch.save(dec.state_dict(), os.path.join(save_path, 'decoder_iter{}.pth'.format(iteration)))
                    
                # if iteration % 1000 == 0:
                    # print("\n[iter{}] gaussian number {}".format(iteration, gaussians._scaling.shape[0]))
                
                # Densification
                if iteration < opt.densify_until_iter: #only prune for normal view but not secret
                    visibility_filter_all  = render_pkg["visibility_filter"] & render_pkg["visibility_filter_secret"] & render_pkg["visibility_filter_secret2"]

                    # Update max radii for all views
                    gaussians.max_radii2D[visibility_filter_all] = torch.max(torch.max(torch.max(
                        gaussians.max_radii2D[visibility_filter_all],
                        render_pkg["radii"][visibility_filter_all]),
                        render_pkg["radii_secret"][visibility_filter_all]),
                        render_pkg["radii_secret2"][visibility_filter_all])
                   
                    # Add densification stats for all views
                    gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter_all)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_always_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    dec_optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    dec_optimizer.zero_grad(set_to_none = True)
                    
                del render_pkg
                torch.cuda.empty_cache()
                    
        except RuntimeError as e:
            print("CUDA Error occurred!")
            print("Error message:", str(e))
            print("Last known tensor shapes:")
            print("prompt shape:", prompt.shape)
            print("iteration:", iteration)
            # Print other relevant tensor information
            torch.cuda.empty_cache()
            raise e
def compute_losses(render_pkg, gt_image, gt_secret, gt_secret2, scales_secret_gt, scales_secret2_gt,
                  opt, w_cover, w_secret, w_secret2, w_random):
    """Compute all losses in a memory efficient way"""
    
    Ll1 = l1_loss(render_pkg["render"], gt_image)
    LL1_secret = l1_loss(render_pkg["render_secret"], gt_secret)
    Ll1_random = l1_loss(render_pkg["render_secret_randomtext"], gt_image)
    LL1_secret2 = l1_loss(render_pkg["render_secret2"], gt_secret2)

    #loss for cover
    loss_cover = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(render_pkg["render"], gt_image))
    #loss for secret
    loss_secret = (1.0 - opt.lambda_dssim) * LL1_secret + opt.lambda_dssim * (1.0 - ssim(render_pkg["render_secret"], gt_secret))
    #loss for random
    loss_random = (1.0 - opt.lambda_dssim) * Ll1_random + opt.lambda_dssim * (1.0 - ssim(render_pkg["render_secret_randomtext"], gt_image))
    #loss for secret2
    loss_secret2 = (1.0 - opt.lambda_dssim) * LL1_secret2 + opt.lambda_dssim * (1.0 - ssim(render_pkg["render_secret2"], gt_secret2))
    
    return (w_cover * loss_cover + w_secret * loss_secret + w_random * loss_random + 
            w_secret2 * loss_secret2, 
            loss_cover, loss_secret, loss_random, loss_secret2)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])


def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, use_wandb):

    if use_wandb:
        if loss_obj_3d:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "train_loss_patches/loss_obj_3d": loss_obj_3d.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "iter_time": elapsed, "iter": iteration})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if use_wandb:
                        if idx < 5:
                            wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)]})
                            if iteration == testing_iterations[0]:
                                wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [wandb.Image(gt_image)]})
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test, config['name'] + "/loss_viewpoint - psnr": psnr_test})
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity, "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default= [100, 500, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500, 5_000, 5_500, 6_000, 6_500, 7_000, 7_500, 8_000, 8_500,
                                                                            9_000, 9_500, 10_000]) #[1_000, 7_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")
    parser.add_argument("--save_path", type=str, default=None, help="Path to load the pretrained model")
    parser.add_argument("--all_path", type=str, default=None, help="Path to load the ori model")
    parser.add_argument("--hide_path", type=str, default=None, help="Path to load the hide model")
    parser.add_argument("--secret2_path", type=str, default=None, help="Path to load the hide2 model")
    parser.add_argument("--fealist", type=str, default="scale,opacity,rotation,sh,xyz" ,help='list of features to update')
    parser.add_argument("--w_cover", type=float, default=0.5, help='cover-weight')
    parser.add_argument("--w_secret", type=float, default=0.5, help='secret-weight')
    parser.add_argument("--w_secret2", type=float, default=0.5, help='secret2-weight')
    parser.add_argument("--w_random", type=float, default=0.01, help='random-weight')
    parser.add_argument("--source_path_secret", type=str, default=None, help="Path to load secret dataset")
    parser.add_argument("--source_path_secret2", type=str, default=None, help="Path to load secret2 dataset")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(42)
    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="gaussian-splatting")
        wandb.config.args = args
        wandb.run.name = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    fealist = args.fealist.split(',')
    
    # Generate prompt with frozen CLIP
    with torch.inference_mode():  # Even stronger than no_grad
        prompt = generate_prompt("NobodysGonnaKnow") #change to nobodys gonna know
        prompt = prompt.to(device)
    
        # Ensure prompt has no gradients
        prompt.requires_grad_(False)
        
        prompt2 = generate_prompt("HowWouldTheyFind")
        prompt2 = prompt2.to(device)
        prompt2.requires_grad_(False)
        
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    text_encoder.eval()
    # random_prompts = []
    
    random_prompts = []
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    random_strings = [''.join(random.choices(string.ascii_letters, k=16)) for _ in range(30010)]
    random_prompts = generate_prompt_batch(random_strings, tokenizer, text_encoder).to('cpu')
    end_time.record()
    
    
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
    print(f"Time taken to generate random prompts: {elapsed_time:.2f} seconds")
        
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(prompt, prompt2, random_prompts, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations,
             args.debug_from, args.use_wandb,
             args.save_path,
             args.all_path, args.hide_path, args.secret2_path, device, fealist, args.w_cover, args.w_secret,args.w_secret2,args.w_random,
             args.source_path_secret, args.source_path_secret2)

    # All done
    print("\nTraining complete.")
