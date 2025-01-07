/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

// CUDA头文件和GLM库的包含
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
    // 渲染函数，执行反向传播的计算
    void render(
        const dim3 grid, dim3 block,                  // CUDA网格和块的维度
        const uint2* ranges,                          // 点的范围（每个线程处理的区域）
        const uint32_t* point_list,                   // 点列表
        int W, int H,                                 // 图像的宽度和高度
        const float* bg_color,                       // 背景颜色
        const float2* means2D,                       // 2D均值（点在图像上的位置）
        const float4* conic_opacity,                 // 圆锥透明度（如果适用）
        const float* colors,                         // 点的颜色
        const float* depths,                         // 点的深度信息
        const float* final_Ts,                       // 最终变换矩阵（可能是透视变换等）
        const uint32_t* n_contrib,                   // 每个点的贡献数量
        const float* dL_dpixels,                     // 像素梯度（反向传播的图像部分）
        const float* dL_invdepths,                   // 反向传播深度信息
        float3* dL_dmean2D,                          // 2D均值的梯度
        float4* dL_dconic2D,                         // 2D圆锥梯度
        float* dL_dopacity,                          // 透明度梯度
        float* dL_dcolors,                           // 颜色梯度
        float* dL_dinvdepths);                       // 反深度信息梯度

    // 预处理函数，进行一些准备工作和计算（如光照模型、变换矩阵等）
    void preprocess(
        int P, int D, int M,                         // 输入点数、维度和其他标量参数
        const float3* means,                         // 点的3D位置（均值）
        const int* radii,                            // 点的半径（或其他相关参数）
        const float* shs,                            // 球面谐波（Spherical Harmonics）系数
        const bool* clamped,                         // 是否需要限制（例如对某些点的范围进行限制）
        const float* opacities,                      // 透明度信息
        const glm::vec3* scales,                     // 缩放信息
        const glm::vec4* rotations,                  // 旋转信息
        const float scale_modifier,                  // 缩放调整系数
        const float* cov3Ds,                        // 3D协方差矩阵
        const float* view,                           // 视图矩阵
        const float* proj,                           // 投影矩阵
        const float focal_x, float focal_y,          // 焦距（x和y轴方向）
        const float tan_fovx, float tan_fovy,       // 水平和垂直视场的切线
        const glm::vec3* campos,                     // 相机位置
        const float3* dL_dmean2D,                   // 2D均值的梯度（反向传播的图像部分）
        const float* dL_dconics,                     // 圆锥的梯度
        const float* dL_dinvdepth,                   // 反深度的梯度
        float* dL_dopacity,                          // 透明度梯度
        glm::vec3* dL_dmeans,                       // 均值的梯度
        float* dL_dcolor,                           // 颜色的梯度
        float* dL_dcov3D,                           // 3D协方差的梯度
        float* dL_dsh,                              // 球面谐波系数的梯度
        glm::vec3* dL_dscale,                       // 缩放的梯度
        glm::vec4* dL_drot,                         // 旋转的梯度
        bool antialiasing);                         // 是否进行抗锯齿处理
}

#endif
