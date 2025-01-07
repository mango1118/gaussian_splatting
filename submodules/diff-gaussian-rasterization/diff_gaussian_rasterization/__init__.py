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

# 定义一个函数用于对元组中的张量进行深拷贝，并将其移到CPU
def cpu_deep_copy_tuple(input_tuple):
    # 对元组中的每个元素进行深拷贝
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

# 定义一个函数进行高斯光栅化，调用自定义的光栅化操作
def rasterize_gaussians(
    means3D,                 # 3D空间中的高斯位置（每个高斯在3D中的中心坐标）
    means2D,                 # 2D空间中的高斯位置（每个高斯在图像中的投影坐标）
    sh,                      # 球谐系数（用于光照和表面反射建模）
    colors_precomp,          # 预计算的颜色（用于高斯的着色）
    opacities,               # 高斯的透明度
    scales,                  # 高斯的缩放因子
    rotations,               # 高斯的旋转矩阵或四元数
    cov3Ds_precomp,         # 预计算的协方差矩阵（表示每个高斯的形状）
    raster_settings,         # 光栅化设置（可能包括图像的宽度、高度、视角等）
):
    # 调用自定义操作 _RasterizeGaussians 的 apply 方法来执行光栅化
    return _RasterizeGaussians.apply(
        means3D,               # 3D位置
        means2D,               # 2D位置
        sh,                    # 球谐系数
        colors_precomp,        # 颜色
        opacities,             # 透明度
        scales,                # 缩放因子
        rotations,             # 旋转
        cov3Ds_precomp,       # 协方差矩阵
        raster_settings,       # 光栅化设置
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,               # 高斯的3D中心位置（每个高斯在3D空间的坐标）
        means2D,               # 高斯的2D位置（每个高斯在图像中的投影坐标）
        sh,                    # 球谐系数（用于光照和表面反射建模）
        colors_precomp,        # 预计算的颜色（用于高斯的着色）
        opacities,             # 高斯的透明度
        scales,                # 高斯的缩放因子
        rotations,             # 高斯的旋转矩阵或四元数
        cov3Ds_precomp,       # 预计算的协方差矩阵（表示每个高斯的形状）
        raster_settings,       # 光栅化设置（包括视角、相机位置、图像尺寸等）
    ):

        # 重新构造参数的顺序，以符合C++库的要求
        args = (
            raster_settings.bg,            # 背景图像
            means3D,                       # 3D位置
            colors_precomp,                # 颜色
            opacities,                     # 透明度
            scales,                        # 缩放因子
            rotations,                     # 旋转
            raster_settings.scale_modifier, # 缩放修正因子
            cov3Ds_precomp,               # 协方差矩阵
            raster_settings.viewmatrix,    # 视图矩阵
            raster_settings.projmatrix,    # 投影矩阵
            raster_settings.tanfovx,       # 水平视场角
            raster_settings.tanfovy,       # 垂直视场角
            raster_settings.image_height,  # 图像高度
            raster_settings.image_width,   # 图像宽度
            sh,                            # 球谐系数
            raster_settings.sh_degree,     # 球谐级数
            raster_settings.campos,        # 相机位置
            raster_settings.prefiltered,   # 是否预过滤
            raster_settings.antialiasing,  # 是否抗锯齿
            raster_settings.debug          # 调试标志
        )

        # 调用C++/CUDA光栅化函数
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # 保存反向传播所需的张量
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        
        return color, radii, invdepths  # 返回渲染结果：颜色、半径和反深度

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):
        # 从上下文中恢复需要的变量
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # 将参数重新构造为C++方法所期望的顺序
        args = (
            raster_settings.bg,            # 背景图像
            means3D,                       # 3D位置
            radii,                         # 半径
            colors_precomp,                # 颜色
            opacities,                     # 透明度
            scales,                        # 缩放因子
            rotations,                     # 旋转
            raster_settings.scale_modifier, # 缩放修正因子
            cov3Ds_precomp,               # 协方差矩阵
            raster_settings.viewmatrix,    # 视图矩阵
            raster_settings.projmatrix,    # 投影矩阵
            raster_settings.tanfovx,       # 水平视场角
            raster_settings.tanfovy,       # 垂直视场角
            grad_out_color,                # 渲染结果的颜色梯度
            grad_out_depth,                # 渲染结果的深度梯度
            sh,                            # 球谐系数
            raster_settings.sh_degree,     # 球谐级数
            raster_settings.campos,        # 相机位置
            geomBuffer,                    # 几何缓冲区
            num_rendered,                  # 渲染的数量
            binningBuffer,                 # 分箱缓冲区
            imgBuffer,                     # 图像缓冲区
            raster_settings.antialiasing,  # 是否抗锯齿
            raster_settings.debug          # 调试标志
        )

        # 调用C++/CUDA光栅化反向传播函数，计算相关张量的梯度
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # 返回计算的梯度
        grads = (
            grad_means3D,       # 3D位置的梯度
            grad_means2D,       # 2D位置的梯度
            grad_sh,            # 球谐系数的梯度
            grad_colors_precomp, # 颜色的梯度
            grad_opacities,     # 透明度的梯度
            grad_scales,        # 缩放因子的梯度
            grad_rotations,     # 旋转的梯度
            grad_cov3Ds_precomp, # 协方差矩阵的梯度
            None,               # 光栅化设置不需要梯度
        )

        return grads  # 返回所有需要的梯度


class GaussianRasterizationSettings(NamedTuple):
    """
    用于高斯光栅化操作的设置类。
    
    该类包含了所有与光栅化过程相关的设置，用于定义图像的尺寸、相机参数、视场角、光照模型等。
    """

    image_height: int  # 图像的高度（单位：像素）
    image_width: int   # 图像的宽度（单位：像素）
    
    tanfovx: float     # 水平视场角的切线值，定义了相机的视场范围（水平）
    tanfovy: float     # 垂直视场角的切线值，定义了相机的视场范围（垂直）
    
    bg: torch.Tensor   # 背景图像，用于渲染时的背景填充。类型为 `torch.Tensor`
    
    scale_modifier: float  # 缩放修正因子，用于调整高斯模型的大小
    
    viewmatrix: torch.Tensor  # 视图矩阵（相机的变换矩阵），用于从世界坐标系转换到视图坐标系
    
    projmatrix: torch.Tensor  # 投影矩阵（相机的投影矩阵），用于将视图坐标系转换到裁剪空间
    
    sh_degree: int  # 球谐函数的阶数，用于表面反射和光照建模
    
    campos: torch.Tensor  # 相机位置，定义了相机在3D空间中的位置
    
    prefiltered: bool  # 是否使用预过滤的着色器结果，优化渲染效率
    
    debug: bool  # 调试模式标志，启用时可以输出调试信息
    
    antialiasing: bool  # 是否启用抗锯齿，改善图像质量，避免图像锯齿问题


class GaussianRasterizer(nn.Module):
    """
    高斯光栅化器，负责执行基于高斯分布的渲染过程。
    """
    def __init__(self, raster_settings):
        """
        初始化高斯光栅化器。
        
        参数:
            raster_settings (GaussianRasterizationSettings): 渲染过程中使用的设置，包括图像大小、相机视图、投影矩阵等参数。
        """
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        """
        标记可见的点，基于视锥体剔除（Frustum Culling）确定哪些点在相机视野内。
        
        参数:
            positions (torch.Tensor): 点的位置，通常是一个 Nx3 的张量，其中 N 是点的数量。
        
        返回:
            torch.Tensor: 一个布尔张量，标记每个点是否在视野内。
        """
        with torch.no_grad():  # 不计算梯度
            raster_settings = self.raster_settings
            # 调用 C++/CUDA 的 mark_visible 函数，进行视锥体剔除
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,  # 视图矩阵
                raster_settings.projmatrix   # 投影矩阵
            )
        
        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):
        """
        前向传播方法，执行高斯光栅化过程。

        参数:
            means3D (torch.Tensor): 点的3D位置，形状为 (P, 3)，其中 P 是点的数量。
            means2D (torch.Tensor): 点的2D位置，形状为 (P, 2)，用于存储光栅化后的2D坐标。
            opacities (torch.Tensor): 点的透明度，形状为 (P,)。
            shs (torch.Tensor, 可选): 球谐函数系数，用于光照建模，形状为 (P, SH_degree)，如果不提供则使用预计算的颜色。
            colors_precomp (torch.Tensor, 可选): 预计算的颜色，形状为 (P, NUM_CHANNELS)，如果不提供则使用球谐函数。
            scales (torch.Tensor, 可选): 点的缩放因子，形状为 (P,)，用于调整高斯大小。
            rotations (torch.Tensor, 可选): 点的旋转，形状为 (P, 4)，表示四元数旋转。
            cov3D_precomp (torch.Tensor, 可选): 预计算的三维协方差矩阵，形状为 (P, 6)，用于表示高斯分布的协方差。

        返回:
            tuple: 返回渲染后的图像、半径、逆深度等结果：
                - color (torch.Tensor): 渲染后的颜色图像，形状为 (NUM_CHANNELS, image_height, image_width)
                - radii (torch.Tensor): 每个点的半径，形状为 (P,)
                - invdepths (torch.Tensor): 渲染后的逆深度图像，形状为 (1, image_height, image_width)
        """
        
        raster_settings = self.raster_settings

        # 验证输入参数的有效性，确保只提供一个 SHs 或者预计算颜色
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide exactly one of either SHs or precomputed colors!')
        
        # 验证输入参数，确保提供了有效的缩放/旋转或预计算的协方差
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        # 如果没有提供 SHs 或者预计算颜色，初始化为空张量
        if shs is None:
            shs = torch.Tensor([])  # 球谐函数的系数
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])  # 预计算的颜色

        # 如果没有提供缩放、旋转或协方差，初始化为空张量
        if scales is None:
            scales = torch.Tensor([])  # 缩放因子
        if rotations is None:
            rotations = torch.Tensor([])  # 旋转
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])  # 预计算的协方差

        # 调用 C++/CUDA 高斯光栅化函数进行渲染
        return rasterize_gaussians(
            means3D,  # 3D 点位置
            means2D,  # 2D 点位置
            shs,       # 球谐函数系数
            colors_precomp,  # 预计算的颜色
            opacities,  # 透明度
            scales,     # 缩放因子
            rotations,  # 旋转
            cov3D_precomp,  # 预计算的协方差
            raster_settings,  # 渲染设置
        )
