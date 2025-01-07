#pragma once
#include <torch/extension.h>  // PyTorch C++扩展接口
#include <cstdio>
#include <tuple>
#include <string>

// 函数原型：CUDA加速的高斯点云渲染（前向计算）
// 参数：
// - background：背景图像，torch::Tensor类型
// - means3D：每个高斯的3D中心点，torch::Tensor类型
// - colors：高斯的颜色，torch::Tensor类型
// - opacity：每个高斯的透明度，torch::Tensor类型
// - scales：每个高斯的缩放因子，torch::Tensor类型
// - rotations：每个高斯的旋转矩阵，torch::Tensor类型
// - scale_modifier：缩放调整因子，float类型
// - cov3D_precomp：高斯的预计算协方差，torch::Tensor类型
// - viewmatrix：视图矩阵，torch::Tensor类型
// - projmatrix：投影矩阵，torch::Tensor类型
// - tan_fovx：水平视场的正切值，float类型
// - tan_fovy：垂直视场的正切值，float类型
// - image_height：图像的高度，int类型
// - image_width：图像的宽度，int类型
// - sh：球谐函数系数，torch::Tensor类型
// - degree：球谐函数的阶数，int类型
// - campos：相机的位置，torch::Tensor类型
// - prefiltered：是否预滤波，bool类型
// - antialiasing：是否使用抗锯齿，bool类型
// - debug：是否开启调试模式，bool类型
// 返回：一个包含7个Tensor的元组，表示渲染结果、深度图和其他相关数据
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug);

// 函数原型：CUDA加速的高斯点云渲染（反向传播计算）
// 参数：
// - background：背景图像，torch::Tensor类型
// - means3D：每个高斯的3D中心点，torch::Tensor类型
// - radii：每个高斯的半径，torch::Tensor类型
// - colors：每个高斯的颜色，torch::Tensor类型
// - scales：每个高斯的缩放因子，torch::Tensor类型
// - opacities：每个高斯的透明度，torch::Tensor类型
// - rotations：每个高斯的旋转矩阵，torch::Tensor类型
// - scale_modifier：缩放调整因子，float类型
// - cov3D_precomp：高斯的预计算协方差，torch::Tensor类型
// - viewmatrix：视图矩阵，torch::Tensor类型
// - projmatrix：投影矩阵，torch::Tensor类型
// - tan_fovx：水平视场的正切值，float类型
// - tan_fovy：垂直视场的正切值，float类型
// - dL_dout_color：颜色输出的梯度，torch::Tensor类型
// - dL_dout_invdepth：深度输出的梯度，torch::Tensor类型
// - sh：球谐函数系数，torch::Tensor类型
// - degree：球谐函数的阶数，int类型
// - campos：相机的位置，torch::Tensor类型
// - geomBuffer：几何数据缓存，torch::Tensor类型
// - R：渲染后的高斯数量，int类型
// - binningBuffer：分箱数据缓存，torch::Tensor类型
// - imageBuffer：图像数据缓存，torch::Tensor类型
// - antialiasing：是否使用抗锯齿，bool类型
// - debug：是否开启调试模式，bool类型
// 返回：一个包含8个Tensor的元组，表示梯度计算的结果
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& opacities,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug);

// 函数原型：根据3D中心点、视图矩阵和投影矩阵标记高斯点是否可见
// 参数：
// - means3D：每个高斯的3D中心点，torch::Tensor类型
// - viewmatrix：视图矩阵，torch::Tensor类型
// - projmatrix：投影矩阵，torch::Tensor类型
// 返回：一个Tensor，表示每个高斯点是否可见（True表示可见，False表示不可见）
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);
