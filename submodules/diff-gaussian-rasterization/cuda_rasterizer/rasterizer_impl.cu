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

#include "rasterizer_impl.h"   // 自定义的光栅化实现相关头文件
#include <iostream>             // 用于输入输出流（如打印日志）
#include <fstream>              // 用于文件操作
#include <algorithm>            // 用于常见的算法操作（如排序）
#include <numeric>              // 用于数字计算（如累加）
#include <cuda.h>               // CUDA API
#include "cuda_runtime.h"       // CUDA运行时API
#include "device_launch_parameters.h" // 用于设备的启动参数
#include <cub/cub.cuh>          // CUB库：CUDA高级算法库
#include <cub/device/device_radix_sort.cuh> // CUB设备排序算法（基数排序）
#define GLM_FORCE_CUDA          // 强制启用CUDA支持的GLM（数学库）
#include <glm/glm.hpp>          // GLM：OpenGL数学库

#include <cooperative_groups.h> // 协同组（cooperative groups）头文件，用于高效的线程协同计算
#include <cooperative_groups/reduce.h> // 协同组的归约（reduction）操作
namespace cg = cooperative_groups;  // 简化命名空间为cg

#include "auxiliary.h"          // 辅助函数的头文件
#include "forward.h"            // 向前传播相关的头文件
#include "backward.h"           // 向后传播相关的头文件


// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;  // 计算32位整数的位数（通常为32）
    uint32_t step = msb;           // 步长初始化为msb的值
    while (step > 1)               // 当步长大于1时，继续调整
    {
        step /= 2;                  // 每次将步长减半
        if (n >> msb)               // 检查n的高位是否为1
            msb += step;            // 如果是，msb增加步长
        else
            msb -= step;            // 否则，msb减小步长
    }
    if (n >> msb)                   // 最终检查高位的值
        msb++;                       // 如果是1，则msb增加
    return msb;
}


// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
    const float* orig_points,
    const float* viewmatrix,
    const float* projmatrix,
    bool* present)
{
    // 获取当前线程的索引
    auto idx = cg::this_grid().thread_rank();
    
    // 如果线程的索引超出了需要处理的点数量P，则返回
    if (idx >= P)
        return;

    float3 p_view;
    
    // 使用辅助函数in_frustum进行视锥体内检测，p_view是检测过程中使用的视图坐标
    present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
    int P,
    const float2* points_xy,
    const float* depths,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    int* radii,
    dim3 grid)
{
    // 获取当前线程在网格中的索引
    auto idx = cg::this_grid().thread_rank();
    
    // 如果当前线程的索引超出点的总数P，则返回
    if (idx >= P)
        return;

    // 如果高斯的半径大于0，说明它是可见的
    if (radii[idx] > 0)
    {
        // 查找当前高斯的偏移量，用于写入键值对
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        uint2 rect_min, rect_max;

        // 获取该高斯所在的矩形区域的最小和最大坐标
        getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

        // 遍历该高斯的矩形区域，检查它与哪些瓦片有重叠
        // 对于每个与该高斯重叠的瓦片，生成一个键值对
        // 键是 | tile ID | depth |，值是高斯的 ID
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                // 计算当前瓦片的 ID
                uint64_t key = y * grid.x + x;
                key <<= 32;  // 将瓦片 ID 移到高位
                key |= *((uint32_t*)&depths[idx]);  // 将该高斯的深度值放入低位

                // 存储键值对
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;

                // 更新偏移量
                off++;
            }
        }
    }
}


// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
    // 获取当前线程在网格中的索引
    auto idx = cg::this_grid().thread_rank();
    
    // 如果当前线程的索引大于高斯点总数L，则跳过当前线程
    if (idx >= L)
        return;

    // 从键中读取瓦片 ID，键的高位存储的是瓦片 ID，低位是深度信息
    uint64_t key = point_list_keys[idx];
    uint32_t currtile = key >> 32;  // 提取当前高斯所在瓦片的 ID

    // 如果是第一个元素，设置当前瓦片的开始位置
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        // 获取前一个高斯所在瓦片的 ID
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        
        // 如果当前瓦片与前一个瓦片不同，更新前一个瓦片的结束位置
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;  // 更新前一个瓦片的结束位置
            ranges[currtile].x = idx;  // 设置当前瓦片的开始位置
        }
    }

    // 如果是最后一个元素，设置当前瓦片的结束位置
    if (idx == L - 1)
        ranges[currtile].y = L;
}


// Mark Gaussians as visible/invisible, based on view frustum testing
// 通过视锥体测试标记哪些高斯点是可见的
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

// 从chunk缓冲区提取和初始化GeometryState结构
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}
