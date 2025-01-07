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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
// 定义一个函数，通过球谐函数的系数来计算高斯椭球的颜色
// 其实实现的功能和utils/sh_utils.py中的函数一样
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
    // 实现的思路参考了 Zhang 等人（2022）在论文 "Differentiable Point-Based Radiance Fields for Efficient View Synthesis" 中的代码
    glm::vec3 pos = means[idx];  // 获取第 idx 个高斯点的 3D 坐标
    glm::vec3 dir = pos - campos;  // 计算相机与高斯点的向量
    dir = dir / glm::length(dir);  // 归一化方向向量

    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;  // 获取该高斯点的球谐系数
    glm::vec3 result = SH_C0 * sh[0];  // 初始结果是球谐函数的 C0 项

    // 如果球谐函数的阶数大于 0，继续计算高阶项
    if (deg > 0)
    {
        float x = dir.x;  // 提取方向向量的分量
        float y = dir.y;
        float z = dir.z;

        // 计算球谐函数的第一阶项
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        // 如果阶数大于 1，继续计算第二阶项
        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            // 计算球谐函数的第二阶项
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];

            // 如果阶数大于 2，继续计算第三阶项
            if (deg > 2)
            {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;  // 加上偏移量，确保颜色不为负值

    // RGB 颜色是限定在正值范围的。如果颜色值被夹紧（小于 0），
    // 我们需要记录这种情况，以便在反向传播时处理。
    clamped[3 * idx + 0] = (result.x < 0);  // 如果结果的红色分量小于 0，设置为 1（表示夹紧）
    clamped[3 * idx + 1] = (result.y < 0);  // 如果结果的绿色分量小于 0，设置为 1（表示夹紧）
    clamped[3 * idx + 2] = (result.z < 0);  // 如果结果的蓝色分量小于 0，设置为 1（表示夹紧）

    // 返回一个非负的颜色值，如果小于 0 则设置为 0
    return glm::max(result, 0.0f);
}

// 计算二维高斯分布的协方差矩阵
// 输入参数：
// - mean: 高斯点的三维均值
// - focal_x, focal_y: 相机的焦距（x 和 y 方向的焦距）
// - tan_fovx, tan_fovy: 相机在水平和垂直方向的视场角的切线值
// - cov3D: 三维高斯分布的协方差矩阵
// - viewmatrix: 世界坐标系到相机坐标系的投影矩阵
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
    // 通过自定义函数 transformPoint4x3 将三维空间中的高斯点 mean 转换到相机坐标系中的位置 t
    // 该函数将 mean 乘以 viewmatrix 进行坐标变换，得到该点在相机坐标系下的坐标
    float3 t = transformPoint4x3(mean, viewmatrix);

    // 限制水平和垂直方向的视野范围为 1.3 倍的视场角
    // 这一步是为了防止点投影到视图平面时超出合理的视野范围
    const float limx = 1.3f * tan_fovx;  // 限制 x 方向的视野范围
    const float limy = 1.3f * tan_fovy;  // 限制 y 方向的视野范围

    // 计算相机坐标系中 x 和 y 坐标与 z 坐标的比值
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;

    // 限制 t.x 和 t.y 的值，使其落在视野范围内
    t.x = min(limx, max(-limx, txtz)) * t.z;  // 限制 x 坐标
    t.y = min(limy, max(-limy, tytz)) * t.z;  // 限制 y 坐标

    // 构建雅可比矩阵 J，用于将三维坐标系中的点投影到二维视图平面
    // 这个雅可比矩阵与相机的焦距和点的深度（z 值）相关
    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),  // x 方向的变换
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),  // y 方向的变换
        0, 0, 0);  // z 方向没有变化

    // 从视图矩阵中提取相机坐标系的变换矩阵 W（3x3 部分）
    // 这个矩阵将点从世界坐标系转换到相机坐标系
    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    // 计算最终的变换矩阵 T = W * J
    // T 矩阵用于将三维协方差矩阵变换到二维视图平面
    glm::mat3 T = W * J;

    // 使用传入的三维协方差矩阵 cov3D 构造 Vrk 矩阵
    // Vrk 矩阵表示三维高斯分布的协方差矩阵
    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],  // 第一行
        cov3D[1], cov3D[3], cov3D[4],  // 第二行
        cov3D[2], cov3D[4], cov3D[5]); // 第三行

    // 计算二维协方差矩阵 cov = T^T * Vrk * T
    // 这里 T^T 是 T 矩阵的转置，Vrk 是三维协方差矩阵
    // 通过这个矩阵运算可以得到二维投影下的协方差矩阵
    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    // 返回二维协方差矩阵的前两项，cov[0][0], cov[0][1], cov[1][1]
    // 这三个值构成了二维协方差矩阵的核心部分
    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// 计算三维协方差矩阵的函数
// 输入参数：
// - scale: 缩放因子（表示高斯分布的大小变化）
// - mod: 一个调节因子，用于调整缩放（可以理解为一个全局缩放系数）
// - rot: 旋转因子（四元数形式，表示旋转）
// - cov3D: 输出的三维协方差矩阵（预先定义的数组，用于存储结果）
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// 创建缩放矩阵 S，初始值为单位矩阵
	glm::mat3 S = glm::mat3(1.0f);

	// 根据缩放因子（`scale`）和调节因子（`mod`）设置缩放矩阵的对角元素
	S[0][0] = mod * scale.x;  // x 方向的缩放
	S[1][1] = mod * scale.y;  // y 方向的缩放
	S[2][2] = mod * scale.z;  // z 方向的缩放

	// 归一化四元数（虽然四元数 `rot` 传入时没有做归一化，但这里假设它是有效的四元数）
	glm::vec4 q = rot; // 取旋转四元数
	// 从四元数中提取旋转分量
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// 根据四元数计算旋转矩阵 R
	// 旋转矩阵是基于四元数计算的，下面的公式源自四元数到旋转矩阵的转换公式
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// 最终的变换矩阵 M 是缩放矩阵 S 和旋转矩阵 R 的乘积
	glm::mat3 M = S * R;

	// 计算三维世界坐标系下的协方差矩阵 Sigma
	// 协方差矩阵 Sigma 是矩阵 M 的转置矩阵与矩阵 M 相乘得到的
	glm::mat3 Sigma = glm::transpose(M) * M;

	// 协方差矩阵是对称的，我们只需要存储上三角部分
	// 将协方差矩阵的上三角部分存储到 `cov3D` 数组中
	cov3D[0] = Sigma[0][0];  // Sigma 的 (0, 0) 元素
	cov3D[1] = Sigma[0][1];  // Sigma 的 (0, 1) 元素
	cov3D[2] = Sigma[0][2];  // Sigma 的 (0, 2) 元素
	cov3D[3] = Sigma[1][1];  // Sigma 的 (1, 1) 元素
	cov3D[4] = Sigma[1][2];  // Sigma 的 (1, 2) 元素
	cov3D[5] = Sigma[2][2];  // Sigma 的 (2, 2) 元素
}

// 每个高斯体（点）在光栅化前的预处理步骤。
// 这个内核会执行多项任务，包括协方差计算、颜色转换和视锥裁剪。
// 输入：
// - P: 高斯点的总数量。
// - D, M: 球谐函数的阶数和系数数量。
// - orig_points: 原始高斯点的位置数据。
// - scales: 每个高斯体的缩放因子。
// - scale_modifier: 缩放调整因子。
// - rotations: 旋转因子（四元数形式）。
// - opacities: 不透明度值。
// - shs: 球谐函数系数。
// - clamped: 用于标记是否需要裁剪的标志。
// - cov3D_precomp: 预计算的三维协方差矩阵。
// - colors_precomp: 预计算的颜色数据。
// - viewmatrix: 相机视图矩阵。
// - projmatrix: 投影矩阵。
// - cam_pos: 相机位置。
// - W, H: 图像的宽度和高度。
// - tan_fovx, tan_fovy: 视野的切线值。
// - focal_x, focal_y: 相机的焦距。
// - radii: 存储每个高斯点的半径。
// - points_xy_image: 存储每个点在屏幕空间的坐标。
// - depths: 存储每个点的深度信息。
// - cov3Ds: 存储三维协方差矩阵。
// - rgb: 存储每个点的RGB颜色。
// - conic_opacity: 存储椭圆形高斯的反协方差矩阵和不透明度。
// - grid: 用于计算网格的维度。
// - tiles_touched: 用于记录每个高斯体影响的网格区域。
// - prefiltered: 是否使用预过滤的视锥裁剪。
// - antialiasing: 是否启用抗锯齿。
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	// 获取当前线程的下标
	auto idx = cg::this_grid().thread_rank();
	// 如果当前线程下标大于点的总数P，则跳出，不进行处理
	if (idx >= P)
		return;

	// 首先，初始化了一些变量，包括半径（radii）和触及到的瓦片数量（tiles_touched）。
	radii[idx] = 0;  // 初始化半径为0
	tiles_touched[idx] = 0;  // 初始化触及图块数量为0

	// 使用 in_frustum 函数进行近裁剪，如果点在视锥体之外，则退出。
	// p_view 用于存储经过视图变换的点
	float3 p_view;
	// 使用函数 `in_frustum` 判断点是否在视锥内，如果不在则跳过
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// 对原始点进行投影变换，计算其在屏幕上的坐标。
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	// 将原始点投影到齐次坐标中
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// 将齐次坐标转换为屏幕空间坐标
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// 如果已经预先计算了三维协方差矩阵，则直接使用
	// 否则根据缩放和旋转因子计算三维协方差矩阵
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		// 使用预先计算的协方差矩阵
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		// 如果没有预处理，则根据缩放和旋转因子计算协方差矩阵
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// 根据3D协方差矩阵、焦距和视锥体矩阵，计算2D屏幕空间的协方差矩阵。
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// 为了抗锯齿和数值稳定性，调整高斯卷积缩放
	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	// 如果启用了抗锯齿，计算抗锯齿缩放因子
	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // 数值稳定性处理

	// 计算协方差矩阵的逆矩阵，基于EWA算法（精细加权平均算法）
	const float det = det_cov_plus_h_cov;
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// 计算2D协方差矩阵的特征值，用于计算屏幕空间的范围，以确定与之相交的瓦片。
	// 计算高斯分布的半径（基于协方差矩阵的特征值）
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// 根据特征值计算高斯分布的半径
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// 将投影后的坐标从标准化设备坐标（NDC）转换为像素坐标
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	// 获取该高斯分布覆盖的矩形区域
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	// 如果矩形面积为0，表示该高斯分布在屏幕上不可见，跳过处理
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// 如果颜色已经预先计算，则直接使用预计算的颜色
	if (colors_precomp == nullptr)
	{
		// 否则，基于球谐函数系数计算颜色
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// 存储一些对后续步骤有用的数据
	// 存储该高斯点的深度值
	depths[idx] = p_view.z;
	// 存储该高斯点的半径
	radii[idx] = my_radius;
	// 存储该点在屏幕空间的坐标
	points_xy_image[idx] = point_image;
	// 将逆二维协方差矩阵和不透明度存储在一个结构体中
	float opacity = opacities[idx];
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };

	// 存储该高斯点影响的图块数量
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 最核心的部分，用于渲染光栅化平面
/*	
	核心思路：
	1、通过计算当前线程所属的 tile 的范围，确定当前线程要处理的像素区域。
	2、判断当前线程是否在有效像素范围内，如果不在，则将 done 设置为 true，表示该线程不执行渲染操作。
	3、使用 __syncthreads_count 函数，统计当前块内 done 变量为 true 的线程数，如果全部线程都完成，跳出循环。
	4、在每个迭代中，从全局内存中收集每个线程块对应的范围内的数据，包括点的索引、2D 坐标和锥体参数透明度。
	5、对当前线程块内的每个点，进行基于锥体参数的渲染，计算贡献并更新颜色。
	6、所有线程处理完毕后，将渲染结果写入 final_T、n_contrib 和 out_color。
 */
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges, //包含了每个范围的起始和结束索引的数组。
	const uint32_t* __restrict__ point_list, //包含了点的索引的数组。
	int W, int H, //图像的宽度和高度。
	const float2* __restrict__ points_xy_image, //包含每个点在屏幕上的坐标的数组。
	const float* __restrict__ features, //包含每个点的颜色信息的数组。
	const float4* __restrict__ conic_opacity, //包含每个点的锥体参数和透明度信息的数组。
	float* __restrict__ final_T, //用于存储每个像素的最终颜色的数组。（多个叠加？）
	uint32_t* __restrict__ n_contrib, //用于存储每个像素的贡献计数的数组。
	const float* __restrict__ bg_color, //如果提供了背景颜色，将其作为背景。
	float* __restrict__ out_color, //存储最终渲染结果的数组。
    const float* __restrict__ depths,
    float* __restrict__ invdepth)
{
    // Identify current tile and associated min/max pixel range.
    // 1.确定当前像素范围：
	// 这部分代码用于确定当前线程块要处理的像素范围，包括 pix_min 和 pix_max，并计算当前线程对应的像素坐标 pix。
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;  // 水平方向上需要多少个 block
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };  // 当前 block 的左上角像素坐标
    int2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };  // 当前 block 的右下角像素坐标
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };  // 当前处理的像素位置
    uint32_t pix_id = W * pix.y + pix.x;  // 当前像素在图像中的 ID
    float2 pixf = { (float)pix.x, (float)pix.y };  // 当前像素的浮动坐标

	// 2.判断当前线程是否在有效像素范围内：
	// 根据像素坐标判断当前线程是否在有效的图像范围内，如果不在，则将 done 设置为 true，表示该线程无需执行渲染操作。
    bool inside = pix.x < W && pix.y < H;  // 判断当前像素是否在有效范围内
    bool done = !inside;  // 如果像素在外部，标记为完成

    // Load start/end range of IDs to process in bit sorted list.
	// 3.加载点云数据处理范围：
	// 这部分代码加载当前线程块要处理的点云数据的范围，即 ranges 数组中对应的范围，并计算点云数据的迭代批次 rounds 和总共要处理的点数 toDo。
    // 获取当前 block 需要处理的高斯点范围
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);  // 需要的轮次
    int toDo = range.y - range.x;  // 当前需要处理的点数

    // Allocate storage for batches of collectively fetched data.
	// 4. 初始化共享内存：
	// 分别定义三个共享内存数组，用于在每个线程块内共享数据。
    // 为每个 batch 分配存储空间
    __shared__ int collected_id[BLOCK_SIZE];  // 存储当前 batch 中的点 ID
    __shared__ float2 collected_xy[BLOCK_SIZE];  // 存储当前 batch 中的点的坐标
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];  // 存储当前 batch 中的点的反协方差矩阵和不透明度

    // Initialize helper variables
	// 5.初始化渲染相关变量：
	// 初始化渲染所需的一些变量，包括当前像素颜色 C、贡献者数量等。
    // 初始化一些帮助变量
    float T = 1.0f;  // 初始透明度（阿尔法混合的 T）
    uint32_t contributor = 0;  // 当前贡献者计数器
    uint32_t last_contributor = 0;  // 最后一个贡献者的计数
    float C[CHANNELS] = { 0 };  // 存储当前像素的颜色值（每个通道）
    float expected_invdepth = 0.0f;  // 存储反深度

    // Iterate over batches until all done or range is complete
	// 6.迭代处理点云数据：
	// 在每个迭代中，处理一批点云数据。内部循环迭代每个点，进行基于锥体参数的渲染计算，并更新颜色信息。
	// Iterate over batches until all done or range is complete
    // 处理每个 batch，直到所有点都被处理完或者结束
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // End if entire block votes that it is done rasterizing
		// 检查是否所有线程块都已经完成渲染：
		// 通过 __syncthreads_count 统计已经完成渲染的线程数，如果整个线程块都已完成，则跳出循环。
        // 如果 block 中所有线程都完成了光栅化，跳出循环
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Collectively fetch per-Gaussian data from global to shared memory
        // 将数据从全局内存加载到共享内存
		// 每个线程通过索引 progress 计算要加载的点云数据的索引 coll_id，然后从全局内存中
		// 加载到共享内存 collected_id、collected_xy 和 collected_conic_opacity 中。block.sync() 确保所有线程都加载完成。
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            // 获取当前点的信息
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        }
        block.sync();  // 同步所有线程

        // Iterate over current batch
		// 迭代处理当前批次的点云数据：
        // 遍历当前的 batch 中的每个点
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)  //在当前批次的循环中，每个线程处理一条点云数据。
        {
            contributor++;  // 当前贡献者数加 1

			// 计算当前点的投影坐标与锥体参数的差值：
			// 计算当前点在屏幕上的坐标 xy 与当前像素坐标 pixf 的差值，并使用锥体参数计算 power。
            float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            // 获取该点的反协方差矩阵和不透明度
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            // 如果 power 大于 0，则跳过此点，因为它不在高斯分布的影响范围内
            if (power > 0.0f)
                continue;

			// 计算论文中公式2的 alpha，利用高斯分布的指数衰减函数
            float alpha = min(0.99f, con_o.w * exp(power));
            if (alpha < 1.0f / 255.0f)
                continue;
            float test_T = T * (1 - alpha);
            if (test_T < 0.0001f)
            {
                done = true;
                continue;
            }

            // 累加颜色
			// 使用高斯分布进行渲染计算：更新颜色信息 C。
            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

            if (invdepth)
                expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

            T = test_T;  // 更新透明度

            last_contributor = contributor;  // 更新最后一个贡献者
        }
    }

    // All threads that treat valid pixel write out their final rendering data to the frame and auxiliary buffers.
	//7. 写入最终渲染结果：
	// 如果当前线程在有效像素范围内，则将最终的渲染结果写入相应的缓冲区，包括 final_T、n_contrib 和 out_color。
    // 所有线程都处理有效像素并写入最终渲染数据到输出缓冲区
    if (inside)
    {
        // 更新最终不透明度、贡献者数量和颜色
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];  // 颜色加上背景颜色

        if (invdepth)
            invdepth[pix_id] = expected_invdepth;  // 更新反深度
    }
}

// 渲染函数，负责调用CUDA内核进行图像渲染
void FORWARD::render(
	const dim3 grid, dim3 block, // 网格和线程块的维度
	const uint2* ranges, // 每个block处理的高斯点范围
	const uint32_t* point_list, // 点ID列表
	int W, int H, // 图像的宽度和高度
	const float2* means2D, // 每个点的2D坐标（经过投影后的坐标）
	const float* colors, // 每个点的颜色信息
	const float4* conic_opacity, // 每个点的反协方差矩阵和不透明度
	float* final_T, // 每个像素的最终透明度
	uint32_t* n_contrib, // 每个像素的贡献者数量
	const float* bg_color, // 背景颜色
	float* out_color, // 输出图像的最终颜色
	float* depths, // 深度信息
	float* depth) // 最终深度
{
	// 调用CUDA核函数进行渲染
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges, // 每个block处理的点范围
		point_list, // 点列表
		W, H, // 图像宽度和高度
		means2D, // 2D坐标
		colors, // 点颜色
		conic_opacity, // 反协方差矩阵和不透明度
		final_T, // 最终透明度
		n_contrib, // 贡献者数量
		bg_color, // 背景颜色
		out_color, // 输出颜色
		depths, // 深度
		depth); // 最终深度
}

// 预处理函数，负责在渲染前对数据进行处理
void FORWARD::preprocess(
	int P, int D, int M, // 点的数量、维度、特征数量
	const float* means3D, // 每个点的3D坐标
	const glm::vec3* scales, // 每个点的缩放信息
	const float scale_modifier, // 缩放系数
	const glm::vec4* rotations, // 每个点的旋转信息
	const float* opacities, // 每个点的不透明度
	const float* shs, // 球面谐波系数
	bool* clamped, // 是否被限制的标志
	const float* cov3D_precomp, // 预计算的3D协方差矩阵
	const float* colors_precomp, // 预计算的颜色
	const float* viewmatrix, // 视图矩阵
	const float* projmatrix, // 投影矩阵
	const glm::vec3* cam_pos, // 相机位置
	const int W, int H, // 图像宽高
	const float focal_x, float focal_y, // 焦距
	const float tan_fovx, float tan_fovy, // 视场角的切线
	int* radii, // 点的半径
	float2* means2D, // 每个点的2D坐标
	float* depths, // 深度信息
	float* cov3Ds, // 3D协方差矩阵
	float* rgb, // 颜色信息
	float4* conic_opacity, // 反协方差矩阵和不透明度
	const dim3 grid, // 网格维度
	uint32_t* tiles_touched, // 被触及的tile
	bool prefiltered, // 是否进行预过滤
	bool antialiasing) // 是否进行抗锯齿
{
	// 调用CUDA核函数进行预处理
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M, // 点数量、维度、特征数量
		means3D, // 3D坐标
		scales, // 缩放信息
		scale_modifier, // 缩放系数
		rotations, // 旋转信息
		opacities, // 不透明度
		shs, // 球面谐波系数
		clamped, // 是否限制的标志
		cov3D_precomp, // 预计算的3D协方差
		colors_precomp, // 预计算颜色
		viewmatrix, // 视图矩阵
		projmatrix, // 投影矩阵
		cam_pos, // 相机位置
		W, H, // 图像宽高
		tan_fovx, tan_fovy, // 视场角的切线
		focal_x, focal_y, // 焦距
		radii, // 半径
		means2D, // 2D坐标
		depths, // 深度信息
		cov3Ds, // 3D协方差矩阵
		rgb, // RGB颜色
		conic_opacity, // 反协方差矩阵和不透明度
		grid, // 网格
		tiles_touched, // 触及的tiles
		prefiltered, // 是否预过滤
		antialiasing // 是否抗锯齿
	);
}
