/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED  // 如果未定义此宏，则定义之
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED  // 防止重复包含

#include <cuda.h>  // 包含 CUDA 相关库
#include "cuda_runtime.h"  // 包含 CUDA 运行时库
#include "device_launch_parameters.h"  // 包含设备启动参数
#define GLM_FORCE_CUDA  // 启用 GLM 库支持 CUDA
#include <glm/glm.hpp>  // 包含 GLM 数学库

namespace FORWARD  // 定义 FORWARD 命名空间，表示前向光栅化处理
{
    // 预处理
    // 为每个高斯点执行光栅化前的初步处理
    void preprocess(int P, int D, int M,
        const float* orig_points,             // 原始点数据
        const glm::vec3* scales,             // 规模向量
        const float scale_modifier,          // 缩放修正因子
        const glm::vec4* rotations,          // 旋转矩阵
        const float* opacities,              // 透明度
        const float* shs,                    // 球面谐波系数
        bool* clamped,                       // 是否被夹紧的标志
        const float* cov3D_precomp,          // 预计算的 3D 协方差
        const float* colors_precomp,         // 预计算的颜色
        const float* viewmatrix,             // 视图矩阵
        const float* projmatrix,             // 投影矩阵
        const glm::vec3* cam_pos,            // 相机位置
        const int W, int H,                  // 图像的宽度和高度
        const float focal_x, float focal_y,  // 焦距
        const float tan_fovx, float tan_fovy,// 水平和垂直视野的切线
        int* radii,                          // 半径
        float2* points_xy_image,             // 图像中的 2D 点
        float* depths,                       // 深度图
        float* cov3Ds,                       // 3D 协方差
        float* colors,                       // 颜色信息
        float4* conic_opacity,               // 圆锥形透明度
        const dim3 grid,                     // CUDA 网格维度
        uint32_t* tiles_touched,             // 触及的瓦片
        bool prefiltered,                    // 是否预过滤
        bool antialiasing                    // 是否开启抗锯齿
    );

    // 渲染
    // 主光栅化方法
    void render(
        const dim3 grid, dim3 block,            // CUDA 网格和 block 尺寸
        const uint2* ranges,                    // 点范围
        const uint32_t* point_list,             // 点列表
        int W, int H,                           // 图像的宽度和高度
        const float2* points_xy_image,          // 图像中的 2D 点
        const float* features,                  // 特征数据
        const float4* conic_opacity,            // 圆锥透明度
        float* final_T,                         // 最终透明度
        uint32_t* n_contrib,                    // 贡献的数量
        const float* bg_color,                  // 背景颜色
        float* out_color,                       // 输出颜色
        float* depths,                          // 深度信息
        float* depth                            // 单独的深度值
    );
}

#endif  // 结束宏定义，防止重复包含