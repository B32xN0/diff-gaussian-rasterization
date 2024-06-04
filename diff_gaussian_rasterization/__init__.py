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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # TODO Project all grad_means3D to nearest point directions from raster_settings.nearest_points
        k = raster_settings.nearest_points.shape[1]

        vectors2nearest = raster_settings.nearest_points - means3D[:, None, :].expand(-1, k, -1)
        # print("Nearest:", torch.sum(nearest))
        # print("Means3D", torch.sum(means3D))
        # TODO normalize directions
        distance2nearest = torch.linalg.norm(vectors2nearest, dim=-1, keepdim=True).expand(-1, -1, 3)
        directions2nearest = vectors2nearest / torch.clamp(distance2nearest, torch.finfo(torch.float32).eps)
        # directions = directions / (dir_norm + torch.finfo(torch.float32).eps)
        # directions = directions / torch.clamp(dir_norm, 1.)
        # del vectors2nearest

        grad_length = torch.linalg.norm(grad_means3D, dim=-1, keepdim=True).expand(-1, 3)
        # grad_direction = grad_means3D / (grad_length + torch.finfo(torch.float32).eps)
        # grad_direction = grad_means3D / torch.clamp(grad_length, 1.)
        # grad_direction = grad_means3D
        grad_directions = grad_means3D / torch.clamp(grad_length, torch.finfo(torch.float32).eps)

        similarity = torch.sum(directions2nearest * grad_directions[:, None, :].expand(-1, k, -1), -1, keepdim=True)
        # del grad_directions
        torch.clamp(similarity, 0)  # TODO zero?
        similarity = similarity / k

        weighted_directions = similarity.expand(-1, -1, 3) * directions2nearest * torch.minimum(distance2nearest, grad_length[:, None, :].expand(-1, k, -1))
        # print("Grad Max", torch.max(grad_length))
        # print("Nearest Max", torch.max(distance2nearest))
        # weighted_directions = similarity.expand(-1, -1, 3) * directions2nearest * distance2nearest
        # del directions2nearest, distance2nearest, grad_length, similarity, k
        new_gradient = torch.sum(weighted_directions, 1)
        # del weighted_directions

        # new_gradient = new_gradient * torch.min(grad_length, torch.mean(dir_norm, dim=1))
        # new_gradient = new_gradient * (1./float(k)) * grad_length
        # new_gradient = new_gradient * (1./float(k))

        # similarity_normalization = torch.sum(similarity, 1).expand(-1, 3)
        # new_gradient = new_gradient / similarity_normalization

        grad_means3D = new_gradient
        # del new_gradient
        # grad_means3D = torch.zeros_like(grad_means3D)
        # grad_means2D = torch.zeros_like(grad_means2D)

        # torch.cuda.empty_cache()

        # clip scaling
        # with torch.no_grad():
            # boundary = torch.quantile(scales, 0.05)
        ### TODO Latest, uncomment two lines:
        # boundary = 0.01
        # grad_scales = torch.where(scales <= boundary, grad_scales, torch.minimum(grad_scales, -torch.ones_like(grad_scales)))

        # grad_scales = grad_scales_new
        # del boundary

        # torch.cuda.empty_cache()

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    nearest_points : torch.Tensor

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

