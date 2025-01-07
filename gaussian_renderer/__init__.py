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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

# 这段代码是一个用于渲染场景的函数，主要是通过将高斯分布的点投影到2D屏幕上来生成渲染图像。
def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, separate_sh=False,
           override_color=None, use_trained_exp=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    """
        渲染场景。
        参数:
            viewpoint_camera: 相机视角对象，包含视图和投影矩阵等信息。
            pc: 高斯模型（GaussianModel）对象，包含场景中物体的高斯分布信息。
            pipe: 渲染管道，包含渲染时的配置，如反走样、调试信息等。
            bg_color: 背景颜色张量，必须位于GPU上。
            scaling_modifier: 缩放因子，用于调整物体的大小，默认为1.0。
            separate_sh: 是否将球面谐波（SH）特征分离，默认为False。
            override_color: 如果给定，将覆盖颜色计算，默认为None。
            use_trained_exp: 是否使用训练好的曝光值，默认为False。
        返回:
            out: 包含渲染结果、视图空间中的点、可见性过滤信息、半径信息和深度图的字典。
        """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建一个与pc中的xyz坐标形状相同的零张量，用于在计算过程中返回2D（屏幕空间）均值的梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()  # 这段代码是一个用于渲染场景的函数，主要是通过将高斯分布的点投影到2D屏幕上来生成渲染图像。
    except:
        pass

    # Set up rasterization configuration
    # 设置光栅化配置参数
    # 计算视场的 tan 值，这将用于设置光栅化配置。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # 水平视场角的切线值
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # 垂直视场角的切线值

    # 配置光栅化器的设置项
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 图像高度
        image_width=int(viewpoint_camera.image_width),  # 图像宽度
        tanfovx=tanfovx,  # 水平视场切线
        tanfovy=tanfovy,  # 垂直视场切线
        bg=bg_color,  # 背景颜色
        scale_modifier=scaling_modifier,  # 缩放因子
        viewmatrix=viewpoint_camera.world_view_transform,  # 世界视图变换矩阵
        projmatrix=viewpoint_camera.full_proj_transform,  # 完整的投影矩阵
        sh_degree=pc.active_sh_degree,  # 当前活动的球面谐波（SH）度数
        campos=viewpoint_camera.camera_center,  # 相机位置
        prefiltered=False,  # 是否预先过滤
        debug=pipe.debug,  # 是否启用调试模式
        antialiasing=pipe.antialiasing  # 是否启用抗锯齿
    )

    # 创建高斯光栅化器对象
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取高斯模型的3D均值、2D均值和不透明度信息
    means3D = pc.get_xyz  # 物体的3D均值（高斯分布的中心）
    means2D = screenspace_points  # 物体在2D图像中的投影坐标
    opacity = pc.get_opacity  # 物体的透明度

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 如果提供了预计算的3D协方差矩阵，直接使用；否则，将通过光栅化器从缩放和旋转计算。
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:  # 是否计算3D协方差
        cov3D_precomp = pc.get_covariance(scaling_modifier)  # 获取高斯模型的3D协方差
    else:
        scales = pc.get_scaling  # 获取缩放因子
        rotations = pc.get_rotation  # 获取旋转矩阵

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预计算的颜色，则直接使用；否则，如果需要从SH特征计算颜色，则执行计算。
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:  # 是否将SH特征转换为RGB颜色
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)  # 转置并调整SH特征的形状
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0],
                                                                         1))  # 计算相机中心到每个点的方向向量
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)  # 将方向向量归一化
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) # 使用SH特征将方向向量转换为RGB颜色。
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) # RGB颜色的范围限制在0到1之间。
        else:
            if separate_sh:  # 如果需要分离SH特征
                dc, shs = pc.get_features_dc, pc.get_features_rest  # 分离SH特征
            else:
                shs = pc.get_features  # 获取SH特征
    else:
        colors_precomp = override_color  # 使用传入的颜色覆盖计算结果

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 使用高斯光栅化器将可见的高斯模型光栅化到图像中，并获得其在屏幕上的半径
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

    # Apply exposure to rendered image (training only)
    # 如果在训练时使用了预训练曝光，则应用曝光到渲染图像
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)  # 获取预训练的曝光参数
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3,
                                                                                                            3, None,
                                                                                                            None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 对于那些被视锥体裁剪掉或半径为0的高斯点，它们是不可见的，应该排除在外
    rendered_image = rendered_image.clamp(0, 1)  # 将渲染结果限制在[0, 1]区间
    out = {
        "render": rendered_image,  # 渲染后的图像
        "viewspace_points": screenspace_points,  # 视图空间中的点
        "visibility_filter": (radii > 0).nonzero(),  # 可见性过滤，去除半径为0的点
        "radii": radii,  # 各点的半径信息
        "depth": depth_image  # 深度图
    }

    return out
