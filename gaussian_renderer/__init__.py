## Copyright (C) 2023, KeySS
# KeySS research group, https://github.com/RY-Paper/KeySS
# All rights reserved.

# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") # + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}
    
  
@torch.no_grad()
def render_imageonly(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return rendered_image



def render_hide_prompt(prompt, decoder, viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") #+ 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # screenspace_points_secret = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # try:
    #     screenspace_points_secret.retain_grad()
    # except:
    #     pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # means2D_secret = screenspace_points_secret
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-16)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            # sh_objs = pc.get_objects
    else:
        colors_precomp = override_color
        
    
    # for normal user
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    #decoder
    sh = shs.reshape(-1,48)
    # degree_mask = degree_mask.to(device)
    
    # Add small epsilon to prevent division by zero and ensure same device
    feature = torch.cat((
        torch.nn.functional.normalize(pc._scaling, eps=1e-16), 
        torch.nn.functional.normalize(pc._opacity, dim=0, eps=1e-16), 
        torch.nn.functional.normalize(sh, eps=1e-16),
        torch.nn.functional.normalize(pc._rotation, eps=1e-16), 
        torch.nn.functional.normalize(pc._xyz, eps=1e-16)
    ), dim=1)

    output_dict = decoder(prompt, feature.unsqueeze(2), pc._scaling, pc._opacity, sh, pc._rotation, pc._xyz)
    #["sh", "rotation", "scale", "opacity", "xyz"]
    if "sh" in output_dict:
        # Output is already masked, just reshape
        shs_secret = output_dict["sh"].reshape(-1, 16, 3) #* degree_mask
    else:
        shs_secret = shs
    
    if "rotation" in output_dict:
        rotations_secret = output_dict["rotation"]
    else:
        rotations_secret = rotations
        
    if "scale" in output_dict:
        scales_secret = output_dict["scale"]
    else:
        scales_secret = scales
        
    if "opacity" in output_dict:
        opacity_secret = output_dict["opacity"]
    else:
        opacity_secret = opacity
        
    if "xyz" in output_dict:
        xyz_secret = output_dict["xyz"]
    else:
        xyz_secret = means3D

    # print(f'scale max: {scales_secret.max().item()}')
    # print(f'scale mean: {scales_secret.mean().item()}')
    # print(f'gs number: {scales_secret.shape[0]}')
    
    # Check all tensors for inf/nan before rasterization
    def check_tensor(name, tensor):
        if isinstance(tensor, torch.Tensor):
            if torch.isinf(tensor).any():
                print(f"Warning: Found inf in {name}")
                print(f"Max: {tensor.max().item()}, Min: {tensor.min().item()}")
                return True
            if torch.isnan(tensor).any():
                print(f"Warning: Found nan in {name}")
                return True
        return False

    # Check all inputs before rasterization
    has_bad_values = False
    has_bad_values |= check_tensor("means3D", xyz_secret)
    has_bad_values |= check_tensor("means2D", means2D)
    has_bad_values |= check_tensor("shs", shs_secret)
    has_bad_values |= check_tensor("opacities", opacity_secret)
    has_bad_values |= check_tensor("scales", scales_secret)
    has_bad_values |= check_tensor("rotations", rotations_secret)

    if has_bad_values:
        # Fix any problematic tensors
        xyz_secret = torch.nan_to_num(xyz_secret, nan=0.0)
        shs_secret = torch.nan_to_num(shs_secret, nan=0.0)
        opacity_secret = torch.nan_to_num(opacity_secret, nan=0.0).clamp(0, 1)
        scales_secret = torch.nan_to_num(scales_secret, nan=1.0)
        rotations_secret = torch.nan_to_num(rotations_secret, nan=0.0)
        if isinstance(means2D, torch.Tensor):
            means2D = torch.nan_to_num(means2D, nan=0.0)

    # Create visibility mask and ensure correct shape
    mask = (radii > 0).float().detach().unsqueeze(-1)
    
    rendered_image_secret, radii_secret = rasterizer(
        means3D=xyz_secret,
        means2D=means2D,
        shs=shs_secret,
        colors_precomp=None,
        opacities=opacity_secret * mask,
        scales=scales_secret * mask,
        rotations=rotations_secret,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            # "render_object": rendered_objects,
            "render_secret": rendered_image_secret,
            "radii_secret": radii_secret,
            "visibility_filter_secret": radii_secret > 0,
            "viewspace_points_secret": screenspace_points,
            "scale_secret":scales_secret,
            "opacity_secret":opacity_secret
            }
    

def render_hide_prompt_hide2(prompt,prompt2, random_prompt, decoder, viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") #+ 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # means2D_secret = screenspace_points_secret
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-16)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            # sh_objs = pc.get_objects
    else:
        colors_precomp = override_color
        
    
    # for normal user
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    #decoder
    sh = shs.view(-1,48)
    
    sh_mask = (shs.abs().sum(dim=2, keepdim=True) > 1e-10).float()
    
    # Apply mask to SH values before normalization
    shs_masked = shs * sh_mask  # N x 16 x 3
    
    # Add small epsilon to prevent division by zero and ensure same device
    feature = torch.cat((
        torch.nn.functional.normalize(pc._scaling, eps=1e-16), 
        torch.nn.functional.normalize(pc._opacity, dim=0, eps=1e-16), 
        torch.nn.functional.normalize(sh, eps=1e-16),
        torch.nn.functional.normalize(pc._rotation, eps=1e-16), 
        torch.nn.functional.normalize(pc._xyz, eps=1e-16)
    ), dim=1)

    output_dict = decoder(prompt, feature.unsqueeze(2), pc._scaling, pc._opacity, sh, pc._rotation, pc._xyz, shs_masked)
    #["sh", "rotation", "scale", "opacity", "xyz"]
    if "sh" in output_dict:
        # Output is already masked, just reshape
        shs_secret = output_dict["sh"].contiguous().view(-1, 16, 3)
    else:
        shs_secret = shs
    
    if "rotation" in output_dict:
        rotations_secret = output_dict["rotation"]
    else:
        rotations_secret = rotations
        
    if "scale" in output_dict:
        scales_secret = output_dict["scale"]
    else:
        scales_secret = scales
        
    if "opacity" in output_dict:
        opacity_secret = output_dict["opacity"]
    else:
        opacity_secret = opacity
        
    if "xyz" in output_dict:
        xyz_secret = output_dict["xyz"]
    else:
        xyz_secret = means3D

    # Create visibility mask and ensure correct shape
    mask = (radii > 0).float().detach().unsqueeze(-1)
    
    rendered_image_secret, radii_secret = rasterizer(
        means3D=xyz_secret,
        means2D=means2D,
        shs=shs_secret,
        colors_precomp=None,
        opacities=opacity_secret * mask,
        scales=scales_secret * mask,
        rotations=rotations_secret,
        cov3D_precomp=None)
    
    output_dict2 = decoder(prompt2, feature.unsqueeze(2), pc._scaling, pc._opacity, sh, pc._rotation, pc._xyz, shs_masked)
    if "sh" in output_dict2:
        # Output is already masked, just reshape
        shs_secret2 = output_dict2["sh"].contiguous().view(-1, 16, 3)
    else:
        shs_secret2 = shs
    if "rotation" in output_dict2:
        rotations_secret2 = output_dict2["rotation"]
    else:
        rotations_secret2 = rotations
        
    if "scale" in output_dict2:
        scales_secret2 = output_dict2["scale"]
    else:
        scales_secret2 = scales
    if "opacity" in output_dict2:
        opacity_secret2 = output_dict2["opacity"]
    else:
        opacity_secret2 = opacity
    if "xyz" in output_dict2:
        xyz_secret2 = output_dict2["xyz"]
    else:
        xyz_secret2 = means3D
        
    rendered_image_secret2, radii_secret2 = rasterizer(
        means3D=xyz_secret2,
        means2D=means2D,
        shs=shs_secret2,
        colors_precomp=None,
        opacities=opacity_secret2 * mask,
        scales=scales_secret2 * mask,
        rotations=rotations_secret2,
        cov3D_precomp=None)
    
    output_dict_randomtext = decoder(random_prompt, feature.unsqueeze(2), pc._scaling, pc._opacity, sh, pc._rotation, pc._xyz, shs_masked)
    if "sh" in output_dict_randomtext:
        # Output is already masked, just reshape
        shs_secret_randomtext = output_dict_randomtext["sh"].contiguous().view(-1, 16, 3)
    else:
        shs_secret_randomtext = shs
    if "rotation" in output_dict_randomtext:
        rotations_secret_randomtext = output_dict_randomtext["rotation"]
    else:
        rotations_secret_randomtext = rotations
    if "scale" in output_dict_randomtext:
        scales_secret_randomtext = output_dict_randomtext["scale"]
    else:
        scales_secret_randomtext = scales
    if "opacity" in output_dict_randomtext:
        opacity_secret_randomtext = output_dict_randomtext["opacity"]
    else:
        opacity_secret_randomtext = opacity
    if "xyz" in output_dict_randomtext:
        xyz_secret_randomtext = output_dict_randomtext["xyz"]
    else:
        xyz_secret_randomtext = means3D
    rendered_image_secret_randomtext, _ = rasterizer(
        means3D=xyz_secret_randomtext,
        means2D=means2D,
        shs=shs_secret_randomtext,
        colors_precomp=None,
        opacities=opacity_secret_randomtext * mask,
        scales=scales_secret_randomtext * mask,
        rotations=rotations_secret_randomtext,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            # "render_object": rendered_objects,
            "render_secret": rendered_image_secret,
            "radii_secret": radii_secret,
            "visibility_filter_secret": radii_secret > 0,
            "render_secret2": rendered_image_secret2,
            "radii_secret2": radii_secret2,
            "visibility_filter_secret2": radii_secret2 > 0,
            "render_secret_randomtext": rendered_image_secret_randomtext,
            "scales_secret": scales_secret,
            "scales_secret2": scales_secret2
            }