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

#include <math.h>                         // 导入数学库，可能会用于后续的数学计算
#include <torch/extension.h>              // 导入PyTorch扩展接口，允许在C++中调用PyTorch功能
#include <cstdio>                         // 导入C标准库的输入输出函数
#include <sstream>                        // 字符串流库，用于格式化字符串（虽然在此代码中未使用）
#include <iostream>                       // 输入输出流库，通常用于打印调试信息
#include <tuple>                          // 元组库，用于返回多个值
#include <stdio.h>                        // 导入C标准库头文件，常用于文件操作
#include <cuda_runtime_api.h>             // CUDA运行时API，用于GPU操作
#include <memory>                         // 内存管理库
#include "cuda_rasterizer/config.h"       // 自定义配置头文件
#include "cuda_rasterizer/rasterizer.h"  // 自定义光栅化操作相关头文件
#include <fstream>                        // 文件输入输出库
#include <string>                         // 字符串库
#include <functional>                     // 引入函数式编程相关功能（如 `std::function`）

// 定义了一个返回Lambda函数的函数 `resizeFunctional`
// 该函数用来动态调整PyTorch tensor的大小，并返回调整后tensor的原始数据指针
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    // 定义Lambda函数，用来调整tensor的大小并返回数据指针
    auto lambda = [&t](size_t N) {
        // 调整tensor的大小，N为新的大小
        t.resize_({(long long)N});
        
        // 返回tensor数据的原始指针
        // `t.contiguous()` 保证tensor内存是连续的
        // `data_ptr()` 返回tensor的数据指针
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };

    // 返回Lambda函数
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,           // 背景图像，用于光栅化过程中计算合成图像
    const torch::Tensor& means3D,              // 高斯点的3D位置（每个高斯点的 (x, y, z) 坐标）
    const torch::Tensor& colors,               // 高斯点的颜色（RGB值）
    const torch::Tensor& opacity,              // 高斯点的透明度
    const torch::Tensor& scales,               // 高斯点的尺度（控制高斯点的大小）
    const torch::Tensor& rotations,            // 高斯点的旋转矩阵
    const float scale_modifier,                // 尺度修正因子，可能用于调整尺度的影响
    const torch::Tensor& cov3D_precomp,        // 预先计算的3D协方差矩阵（用于高斯点的形状控制）
    const torch::Tensor& viewmatrix,           // 相机的视图矩阵（用于3D到2D的投影）
    const torch::Tensor& projmatrix,           // 投影矩阵（用于视角和投影变换）
    const float tan_fovx,                     // 水平视场角的切线值（用于投影计算）
    const float tan_fovy,                     // 垂直视场角的切线值
    const int image_height,                    // 输出图像的高度
    const int image_width,                     // 输出图像的宽度
    const torch::Tensor& sh,                   // 球面谐波（用于渲染时的光照计算）
    const int degree,                          // 球面谐波的阶数
    const torch::Tensor& campos,               // 相机位置
    const bool prefiltered,                    // 是否进行预过滤（例如颜色预计算）
    const bool antialiasing,                   // 是否开启抗锯齿
    const bool debug)                          // 是否开启调试模式
{
    // 检查输入张量 means3D 的维度是否正确
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    const int P = means3D.size(0);  // 高斯点的数量
    const int H = image_height;     // 图像高度
    const int W = image_width;      // 图像宽度

    // 设置数据类型，int 用于索引，float 用于数值计算
    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    // 初始化输出张量
    torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts); // 输出颜色图像
    torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts); // 输出逆深度图像
    float* out_invdepthptr = nullptr; // 逆深度图像的指针

    // 初始化逆深度张量
    out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
    out_invdepthptr = out_invdepth.data<float>();

    // 初始化高斯点的半径，设置为 0（默认值）
    torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

    // 设置 CUDA 设备
    torch::Device device(torch::kCUDA);

    // 初始化其他需要的缓冲区（光栅化相关的缓冲区）
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));  // 几何数据缓冲区
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device)); // 分箱数据缓冲区
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));    // 图像数据缓冲区

    // 定义动态调整内存大小的函数，用于 CUDA 光栅化处理
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    int rendered = 0;  // 用于保存渲染的高斯点数量
    if (P != 0) { // 如果高斯点数量不为0
        int M = 0;
        if (sh.size(0) != 0) {
            M = sh.size(1); // 球面谐波的阶数
        }

        // 调用光栅化前向处理函数（CUDA 内核）
        rendered = CudaRasterizer::Rasterizer::forward(
            geomFunc, 
            binningFunc, 
            imgFunc, 
            P, degree, M, 
            background.contiguous().data<float>(),
            W, H, 
            means3D.contiguous().data<float>(), 
            sh.contiguous().data_ptr<float>(), 
            colors.contiguous().data<float>(), 
            opacity.contiguous().data<float>(), 
            scales.contiguous().data_ptr<float>(),
            scale_modifier, 
            rotations.contiguous().data_ptr<float>(), 
            cov3D_precomp.contiguous().data<float>(), 
            viewmatrix.contiguous().data<float>(), 
            projmatrix.contiguous().data<float>(),
            campos.contiguous().data<float>(),
            tan_fovx,
            tan_fovy,
            prefiltered,
            out_color.contiguous().data<float>(),
            out_invdepthptr,
            antialiasing,
            radii.contiguous().data<int>(),
            debug
        );
    }

    // 返回包含多个信息的元组
    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,           // 背景图像，用于计算损失函数
    const torch::Tensor& means3D,              // 高斯点的3D位置（每个高斯点的 (x, y, z) 坐标）
    const torch::Tensor& radii,                // 高斯点的半径
    const torch::Tensor& colors,               // 高斯点的颜色（RGB值）
    const torch::Tensor& opacities,            // 高斯点的透明度
    const torch::Tensor& scales,               // 高斯点的尺度（控制高斯点的大小）
    const torch::Tensor& rotations,            // 高斯点的旋转矩阵
    const float scale_modifier,                // 尺度修正因子，可能用于调整尺度的影响
    const torch::Tensor& cov3D_precomp,        // 预计算的3D协方差矩阵（用于高斯点的形状控制）
    const torch::Tensor& viewmatrix,           // 相机的视图矩阵（用于3D到2D的投影）
    const torch::Tensor& projmatrix,           // 投影矩阵（用于视角和投影变换）
    const float tan_fovx,                     // 水平视场角的切线值（用于投影计算）
    const float tan_fovy,                     // 垂直视场角的切线值
    const torch::Tensor& dL_dout_color,        // 颜色损失梯度（输出颜色图像的梯度）
    const torch::Tensor& dL_dout_invdepth,     // 深度损失梯度（输出逆深度图像的梯度）
    const torch::Tensor& sh,                   // 球面谐波（用于光照计算）
    const int degree,                          // 球面谐波的阶数
    const torch::Tensor& campos,               // 相机位置
    const torch::Tensor& geomBuffer,           // 几何数据缓冲区，用于存储在光栅化过程中的几何信息
    const int R,                               // 分辨率相关参数，具体意义依赖于光栅化的实现
    const torch::Tensor& binningBuffer,        // 分箱数据缓冲区
    const torch::Tensor& imageBuffer,          // 图像数据缓冲区
    const bool antialiasing,                   // 是否启用抗锯齿
    const bool debug)                          // 是否开启调试模式
{
    const int P = means3D.size(0);  // 高斯点的数量
    const int H = dL_dout_color.size(1);  // 输出图像的高度
    const int W = dL_dout_color.size(2);  // 输出图像的宽度

    int M = 0;
    if (sh.size(0) != 0) {
        M = sh.size(1);  // 球面谐波的阶数
    }

    // 初始化所有需要计算梯度的张量，初始值为零
    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());  // 高斯点3D位置的梯度
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());  // 高斯点2D位置的梯度
    torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());  // 高斯点颜色的梯度
    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());  // 高斯点的椭圆（或者其他）形状的梯度
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());  // 透明度的梯度
    torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());  // 高斯点3D协方差矩阵的梯度
    torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());  // 球面谐波的梯度
    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());  // 尺度的梯度
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());  // 旋转矩阵的梯度
    torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());  // 逆深度的梯度（可能为空）

    float* dL_dinvdepthsptr = nullptr;
    float* dL_dout_invdepthptr = nullptr;
    if (dL_dout_invdepth.size(0) != 0) {
        dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
        dL_dinvdepths = dL_dinvdepths.contiguous();
        dL_dinvdepthsptr = dL_dinvdepths.data<float>();
        dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
    }

    // 如果高斯点数量不为零，进行反向传播计算
    if (P != 0) {
        // 调用CUDA光栅化反向传播函数
        CudaRasterizer::Rasterizer::backward(P, degree, M, R,
            background.contiguous().data<float>(),
            W, H, 
            means3D.contiguous().data<float>(),
            sh.contiguous().data<float>(),
            colors.contiguous().data<float>(),
            opacities.contiguous().data<float>(),
            scales.data_ptr<float>(),
            scale_modifier,
            rotations.data_ptr<float>(),
            cov3D_precomp.contiguous().data<float>(),
            viewmatrix.contiguous().data<float>(),
            projmatrix.contiguous().data<float>(),
            campos.contiguous().data<float>(),
            tan_fovx,
            tan_fovy,
            radii.contiguous().data<int>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            dL_dout_color.contiguous().data<float>(),
            dL_dout_invdepthptr,
            dL_dmeans2D.contiguous().data<float>(),
            dL_dconic.contiguous().data<float>(),
            dL_dopacity.contiguous().data<float>(),
            dL_dcolors.contiguous().data<float>(),
            dL_dinvdepthsptr,
            dL_dmeans3D.contiguous().data<float>(),
            dL_dcov3D.contiguous().data<float>(),
            dL_dsh.contiguous().data<float>(),
            dL_dscales.contiguous().data<float>(),
            dL_drotations.contiguous().data<float>(),
            antialiasing,
            debug
        );
    }

    // 返回包含各个梯度的元组
    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor& means3D,         // 3D点的位置（P, 3），每个点有 (x, y, z) 坐标
    torch::Tensor& viewmatrix,      // 视图矩阵，用于从世界坐标系转换到相机坐标系
    torch::Tensor& projmatrix)      // 投影矩阵，将3D坐标投影到2D图像平面
{
    const int P = means3D.size(0);  // 获取3D点的数量

    // 初始化一个布尔型张量 'present'，大小为 P，默认值为 False，表示初始时所有点都是不可见的
    torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

    // 如果存在3D点，则调用CUDA光栅化内核来计算哪些点是可见的
    if (P != 0)
    {
        // 调用CUDA光栅化函数，markVisible 内核会根据视图矩阵和投影矩阵计算哪些点是可见的
        CudaRasterizer::Rasterizer::markVisible(
            P,                              // 点的数量
            means3D.contiguous().data<float>(),  // 3D点的位置数据
            viewmatrix.contiguous().data<float>(), // 视图矩阵数据
            projmatrix.contiguous().data<float>(), // 投影矩阵数据
            present.contiguous().data<bool>()      // 返回值，标记每个点是否可见
        );
    }

    return present;  // 返回一个布尔型张量，标记每个3D点是否在视野范围内
}
