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

// 防止头文件重复包含
#pragma once

// 引入标准库和CUDA头文件
#include <iostream>
#include <vector>
#include "rasterizer.h" // 光栅化相关的头文件
#include <cuda_runtime_api.h> // CUDA运行时API

// 定义命名空间，避免名称冲突
namespace CudaRasterizer
{
	// obtain函数模板，负责内存对齐并分配空间
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		// 计算对齐后指针的起始地址
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		// 将偏移后的地址强制转换为T类型的指针
		ptr = reinterpret_cast<T*>(offset);
		// 更新chunk指针，移动到已分配内存后的下一个位置
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	// GeometryState结构体，表示几何状态
	struct GeometryState
	{
		size_t scan_size; // 扫描的大小
		float* depths; // 深度信息
		char* scanning_space; // 扫描空间的指针
		bool* clamped; // 是否被夹紧的标志
		int* internal_radii; // 点的半径
		float2* means2D; // 2D点的坐标
		float* cov3D; // 3D协方差矩阵
		float4* conic_opacity; // 反协方差矩阵和不透明度
		float* rgb; // RGB颜色信息
		uint32_t* point_offsets; // 点偏移量
		uint32_t* tiles_touched; // 被触及的tiles

		// 从chunk内存块中初始化GeometryState
		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	// ImageState结构体，表示图像状态
	struct ImageState
	{
		uint2* ranges; // 每个线程块处理的范围
		uint32_t* n_contrib; // 每个像素的贡献者数量
		float* accum_alpha; // 累积的透明度值

		// 从chunk内存块中初始化ImageState
		static ImageState fromChunk(char*& chunk, size_t N);
	};

	// BinningState结构体，表示分箱状态
	struct BinningState
	{
		size_t sorting_size; // 排序的大小
		uint64_t* point_list_keys_unsorted; // 未排序的点列表键
		uint64_t* point_list_keys; // 排序后的点列表键
		uint32_t* point_list_unsorted; // 未排序的点ID列表
		uint32_t* point_list; // 排序后的点ID列表
		char* list_sorting_space; // 排序空间

		// 从chunk内存块中初始化BinningState
		static BinningState fromChunk(char*& chunk, size_t P);
	};

	// 模板函数，返回所需内存的大小
	template<typename T> 
	size_t required(size_t P)
	{
		// 定义一个指针用于存储内存
		char* size = nullptr;
		// 通过调用fromChunk函数计算所需的内存大小
		T::fromChunk(size, P);
		// 返回分配内存所需的大小，加上128字节的缓冲区
		return ((size_t)size) + 128;
	}
};
