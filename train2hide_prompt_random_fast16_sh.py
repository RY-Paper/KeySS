# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
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
from gaussian_renderer import render, network_gui, render_hide_prompt, render_imageonly
import sys
from scene import Scene, GaussianModel, SceneLoad, SceneLoadBoth
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParamsHide, OptimizationParams
import wandb
from submodules.decoder import decoder_all_forloop as decoder_all
from utils.system_utils import mkdir_p
import random
import numpy as np
import torch.nn.functional as F
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

def generate_prompt_batch(text_prompts, tokenizer, text_encoder):
    # Process in batches of 32 to manage memory
    batch_size = 1000
    num_prompts = len(text_prompts)
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, num_prompts, batch_size):
            batch_prompts = text_prompts[i:i + batch_size]
            
            # Tokenize the batch of text
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length
            ).to('cuda')
            
            # Generate embeddings for batch
            text_embeddings = text_encoder(**inputs).last_hidden_state  # [batch_size, seq_len, hidden_dim]
            pooled_embeddings = text_embeddings.mean(dim=1)  # [batch_size, hidden_dim]
            pooled_embeddings = F.normalize(pooled_embeddings.unsqueeze(-1), dim=1, eps=1e-16)
            
            all_embeddings.append(pooled_embeddings)
            
            # Clean up batch tensors
            del inputs, text_embeddings
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Clean up
    del tokenizer, text_encoder, all_embeddings
    torch.cuda.empty_cache()
    
    return final_embeddings.clone().detach()

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
    
    # Create batched input
    token_ids = torch.tensor(batch_token_ids).to('cuda')
    inputs = {'input_ids': token_ids}
    
    # Generate embeddings in batch
    text_embeddings = text_encoder(**inputs).last_hidden_state
    pooled_embeddings = text_embeddings.mean(dim=1)
    pooled_embeddings = F.normalize(pooled_embeddings.unsqueeze(-1), dim=1, eps=1e-16)
    
    return pooled_embeddings.clone().detach()

def training(prompt, random_prompts, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
             debug_from, use_wandb, save_path: str,
             allply:str, hideply:str, device, fealist, w1, w2, w3,batchsize,
             source_path_secret=None):
    first_iter = 0
    dataset.source_path_secret = source_path_secret
    prepare_output_and_logger(dataset)
    
    # Load the CLIP text encoder and tokenizer
    
    # Get mean scaling from secret gaussians
    gaussians_secret = GaussianModel(dataset.sh_degree, device)
    scene_secret = SceneLoad(dataset, gaussians_secret, device, load_path=allply, shuffle=False)
    # mean_scaling = gaussians_secret.get_scaling.mean().item()
    gaussians_secret.freeze()
    # with torch.no_grad():
    #     scales_secret_gt = gaussians_secret.get_scaling.clone().detach()
    #     opacity_secret_gt = gaussians_secret.get_opacity.clone().detach()
    del scene_secret

    #init decoder
    dec = decoder_all(fealist, batchsize=batchsize)
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

    #gs for train
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = SceneLoadBoth(dataset, gaussians, device) #, load_path=hideply)
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
            mse_cover = F.mse_loss(gt_image, viewpoint_cam.original_image)
            mse_secret = F.mse_loss(gt_secret, viewpoint_cam.original_image)
            if mse_secret < mse_cover:
                viewpoint_cam.gt_secret = viewpoint_cam.original_image.detach().clone().cpu()
                viewpoint_cam.gt_cover = gt_image.cpu()
            else:
                viewpoint_cam.gt_cover = viewpoint_cam.original_image.detach().clone().cpu()
                viewpoint_cam.gt_secret = gt_secret.cpu()
            del viewpoint_cam.original_image
            del gt_image, gt_secret
            torch.cuda.empty_cache()
    
    del gaussians_remain, gaussians_secret, scene.train_cameras
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
    warm_up_iter = 0
    
    for iteration in range(first_iter, opt.iterations + 1):   
        if iteration == 1500:
            a = 1    
        #random prompt
        random_prompt = random_prompts[iteration].to(device)
            # print(random_prompt.mean())
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_hide_prompt(dec, custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam = viewpoint_stack[sequence[iteration]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        try:
            
            render_pkg = render_hide_prompt(random_prompt, dec, viewpoint_cam, gaussians, pipe, background)
            image_random = render_pkg["render_secret"]
                
            render_pkg = render_hide_prompt(prompt, dec, viewpoint_cam, gaussians, pipe, background)
            image_secret, image, viewspace_point_tensor, visibility_filter, radii, visibility_filter_secret, viewspace_point_tensor_secret, radii_secret = (
                render_pkg["render_secret"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"],
                render_pkg["radii"],  render_pkg["visibility_filter_secret"], render_pkg["viewspace_points_secret"],render_pkg["radii_secret"])
            with torch.no_grad():
                gt_image = viewpoint_cam.gt_cover.to(device)
                gt_secret = viewpoint_cam.gt_secret.to(device)
            
            # Loss
            Ll1 = l1_loss(image, gt_image)
            LL1_secret = l1_loss(image_secret,gt_secret)
            Ll1_random = l1_loss(image_random, gt_image)

            loss1 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss2 = (1.0 - opt.lambda_dssim) * LL1_secret + opt.lambda_dssim * (1.0 - ssim(image_secret, gt_secret))
            loss3 = (1.0 - opt.lambda_dssim) * Ll1_random + opt.lambda_dssim * (1.0 - ssim(image_random, gt_image))
            loss = w1 * loss1 + w2 * loss2 + w3 * loss3
                
          
            loss.backward()
            iter_end.record()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 100 == 0:
                    progress_bar.set_postfix({"Loss1": f"{loss1:.{3}f}", "Loss2": f"{loss2:.{3}f}", "Loss3": f"{loss3:.{3}f}", "gs_num":f"{gaussians._scaling.shape[0]}"})
                    progress_bar.update(100)
                if iteration == opt.iterations:
                    progress_bar.close()
                    
                if iteration % 10000==0 and iteration >= 20000:
                    scene.save(iteration, save_path) #save the .ply
                    torch.save(dec.state_dict(), os.path.join(save_path, 'decoder_iter{}.pth'.format(iteration)))
                    
                
                # Densification
                if iteration < opt.densify_until_iter: #only prune for normal view but not secret
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > warm_up_iter:
                        visibility_filter_secret &= visibility_filter
                        gaussians.max_radii2D[visibility_filter_secret] = torch.max(gaussians.max_radii2D[visibility_filter_secret],
                                                                            radii_secret[visibility_filter_secret])
                        gaussians.add_densification_stats(viewspace_point_tensor_secret, visibility_filter_secret)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    dec_optimizer.step()
                    dec_optimizer.zero_grad()

        except RuntimeError as e:
            print("CUDA Error occurred!")
            print("Error message:", str(e))
            print("Last known tensor shapes:")
            print("prompt shape:", prompt.shape)
            print("iteration:", iteration)
            # Print other relevant tensor information
            torch.cuda.empty_cache()
            raise e

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
    parser.add_argument("--fealist", type=str, default="scale,opacity,rotation,sh,xyz" ,help='list of features to update')
    parser.add_argument("--w1", type=float, default=0.5, help='loss1-weight')
    parser.add_argument("--w2", type=float, default=0.5, help='loss2-weight')
    parser.add_argument("--w3", type=float, default=0.01, help='loss3-weight')
    parser.add_argument("--source_path_secret", type=str, default=None, help="Path to load secret dataset")
    parser.add_argument("--batchsize", type=int, default=200000, help="batchsize")

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
        
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    text_encoder.eval()
    
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
    training(prompt,random_prompts, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations,
             args.debug_from, args.use_wandb,
             args.save_path,
             args.all_path, args.hide_path, device, fealist, args.w1, args.w2,args.w3,args.batchsize,
             source_path_secret=args.source_path_secret)

    # All done
    print("\nTraining complete.")
