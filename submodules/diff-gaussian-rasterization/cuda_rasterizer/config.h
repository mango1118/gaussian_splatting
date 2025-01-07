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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED  // 如果未定义此宏，则定义之
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED  // 防止重复包含

#define NUM_CHANNELS 3 // 默认为 3，即 RGB 颜色通道
#define BLOCK_X 16     // CUDA block 的宽度定义为 16
#define BLOCK_Y 16     // CUDA block 的高度定义为 16

#endif  // CUDA_RASTERIZER_CONFIG_H_INCLUDED 结束宏定义