#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// 计算平方
__device__ __forceinline__ float sq(float x) { return x * x; }


// 反向传播：根据球面谐波系数计算每个高斯点的颜色梯度，并将其传播到球面谐波系数和高斯位置（means）上
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
    // 获取当前高斯点的位置（3D坐标）
    glm::vec3 pos = means[idx];
    // 计算高斯点到相机的方向
    glm::vec3 dir_orig = pos - campos;
    // 归一化方向向量，表示视角方向
    glm::vec3 dir = dir_orig / glm::length(dir_orig);

    // 获取该高斯点的球面谐波系数
    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

    // 如果颜色通道被钳制，反向传播时对该通道的梯度为0，即颜色通道的值被限定在一个特定的范围内，防止某些异常值影响计算结果
    glm::vec3 dL_dRGB = dL_dcolor[idx];
    dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;  // 钳制第一个颜色通道
    dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;  // 钳制第二个颜色通道
    dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;  // 钳制第三个颜色通道

    // 初始化梯度：x, y, z方向的颜色梯度
    glm::vec3 dRGBdx(0, 0, 0);
    glm::vec3 dRGBdy(0, 0, 0);
    glm::vec3 dRGBdz(0, 0, 0);
    
    // 获取方向向量的分量
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    // 设置目标位置，用来存储计算得到的球面谐波系数梯度
    glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

    // 计算球面谐波系数对颜色的影响（第一阶）
    float dRGBdsh0 = SH_C0;
    dL_dsh[0] = dRGBdsh0 * dL_dRGB;

    if (deg > 0) {
        // 第二阶球面谐波对颜色的贡献
        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        dL_dsh[1] = dRGBdsh1 * dL_dRGB;
        dL_dsh[2] = dRGBdsh2 * dL_dRGB;
        dL_dsh[3] = dRGBdsh3 * dL_dRGB;

        // 计算颜色梯度对球面谐波系数的影响（方向向量的梯度）
        dRGBdx = -SH_C1 * sh[3];
        dRGBdy = -SH_C1 * sh[1];
        dRGBdz = SH_C1 * sh[2];

        if (deg > 1) {
            // 第二阶球面谐波贡献的更多项
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            // 计算每个球面谐波基函数对颜色的贡献
            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh[4] = dRGBdsh4 * dL_dRGB;
            dL_dsh[5] = dRGBdsh5 * dL_dRGB;
            dL_dsh[6] = dRGBdsh6 * dL_dRGB;
            dL_dsh[7] = dRGBdsh7 * dL_dRGB;
            dL_dsh[8] = dRGBdsh8 * dL_dRGB;

            // 更新颜色梯度对球面谐波系数的贡献
            dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
            dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
            dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

            if (deg > 2) {
                // 第三阶球面谐波对颜色的贡献
                float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                dL_dsh[9] = dRGBdsh9 * dL_dRGB;
                dL_dsh[10] = dRGBdsh10 * dL_dRGB;
                dL_dsh[11] = dRGBdsh11 * dL_dRGB;
                dL_dsh[12] = dRGBdsh12 * dL_dRGB;
                dL_dsh[13] = dRGBdsh13 * dL_dRGB;
                dL_dsh[14] = dRGBdsh14 * dL_dRGB;
                dL_dsh[15] = dRGBdsh15 * dL_dRGB;

                // 更新颜色梯度对球面谐波系数的影响
                dRGBdx += (
                    SH_C3[0] * sh[9] * 3.f * 2.f * xy +
                    SH_C3[1] * sh[10] * yz +
                    SH_C3[2] * sh[11] * -2.f * xy +
                    SH_C3[3] * sh[12] * -3.f * 2.f * xz +
                    SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
                    SH_C3[5] * sh[14] * 2.f * xz +
                    SH_C3[6] * sh[15] * 3.f * (xx - yy));

                dRGBdy += (
                    SH_C3[0] * sh[9] * 3.f * (xx - yy) +
                    SH_C3[1] * sh[10] * xz +
                    SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
                    SH_C3[3] * sh[12] * -3.f * 2.f * yz +
                    SH_C3[4] * sh[13] * -2.f * xy +
                    SH_C3[5] * sh[14] * -2.f * yz +
                    SH_C3[6] * sh[15] * -3.f * 2.f * xy);

                dRGBdz += (
                    SH_C3[1] * sh[10] * xy +
                    SH_C3[2] * sh[11] * 4.f * 2.f * yz +
                    SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
                    SH_C3[4] * sh[13] * 4.f * 2.f * xz +
                    SH_C3[5] * sh[14] * (xx - yy));
            }
        }
    }

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	// 计算颜色对方向向量的梯度
    glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));


	// Account for normalization of direction
	// 反向传播到高斯点的3D位置（方向梯度的影响）
    float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });


	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	// 将位置梯度更新到高斯点位置的梯度
    dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
// 计算 2D 协方差矩阵的反向传播梯度
// (由于长度在前面的内核中作为独立内核启动，后面的反向传播步骤在 preprocess 中包含)
__global__ void computeCov2DCUDA(int P,
	const float3* means,  // 高斯的均值
	const int* radii,  // 高斯的半径
	const float* cov3Ds,  // 每个高斯的 3D 协方差矩阵
	const float h_x, float h_y,  // 像素间距
	const float tan_fovx, float tan_fovy,  // 水平和垂直视场角的正切值
	const float* view_matrix,  // 相机视图矩阵
	const float* opacities,  // 高斯的透明度
	const float* dL_dconics,  // 损失函数相对于共轭矩阵的梯度
	float* dL_dopacity,  // 损失函数相对于透明度的梯度
	const float* dL_dinvdepth,  // 损失函数相对于反深度的梯度（可选）
	float3* dL_dmeans,  // 损失函数相对于高斯均值的梯度
	float* dL_dcov,  // 损失函数相对于 3D 协方差矩阵的梯度
	bool antialiasing)  // 是否启用抗锯齿处理
{
	auto idx = cg::this_grid().thread_rank();  // 当前线程的索引
	if (idx >= P || !(radii[idx] > 0))  // 如果索引超出范围或半径为零，直接返回
		return;

	// 获取当前高斯的 3D 协方差矩阵
	const float* cov3D = cov3Ds + 6 * idx;

	// 获取当前高斯的均值和相关的梯度
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };

	// 将 3D 均值转化到视图空间
	float3 t = transformPoint4x3(mean, view_matrix);
	
	// 计算图像空间的坐标限制
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;

	// 计算视图中的归一化坐标
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	// 根据是否超出视场角来设置梯度乘数
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	// 计算 Jacobian 矩阵 J
	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// 从视图矩阵提取相机的变换矩阵 W
	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	// 从 3D 协方差矩阵提取当前高斯的协方差
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// 计算矩阵 T，用于 2D 协方差的计算
	glm::mat3 T = W * J;

	// 计算 2D 协方差矩阵
	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// 提取 2D 协方差矩阵的相关分量
	float c_xx = cov2D[0][0];
	float c_xy = cov2D[0][1];
	float c_yy = cov2D[1][1];
	
	// 设定平滑处理的常数
	constexpr float h_var = 0.3f;
	float d_inside_root = 0.f;

	// 如果启用了抗锯齿处理
	if(antialiasing)
	{
		// 计算 2D 协方差的行列式
		const float det_cov = c_xx * c_yy - c_xy * c_xy;
		c_xx += h_var;
		c_yy += h_var;
		const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;

		// 计算抗锯齿卷积缩放因子
		const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max 为了数值稳定性
		const float dL_dopacity_v = dL_dopacity[idx];
		const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
		dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;

		// 如果行列式变化过小，则取消卷积影响
		d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
	} 
	else
	{
		c_xx += h_var;
		c_yy += h_var;
	}
	
	// 计算 2D 协方差矩阵元素的梯度
	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	if(antialiasing)
	{
		// 计算 2D 协方差的梯度
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
	}
	
	// 计算 2D 协方差的行列式
	float denom = c_xx * c_yy - c_xy * c_xy;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	// 如果行列式有效，则计算梯度
	if (denom2inv != 0)
	{
		// 损失函数相对于 2D 协方差矩阵的梯度
		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);
	}
	
	// 根据导数计算 3D 协方差矩阵的梯度
	// 将 2D 协方差矩阵的梯度反向传播到 3D 协方差矩阵
	// … (更多的反向传播计算)

	// 最后，更新 dL_dmeans 和 dL_dcov（根据 3D 协方差矩阵的梯度进行更新）
}


// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
// 该函数计算每个高斯的3D协方差矩阵的反向传播梯度
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
    // 步骤 1: 重新计算3D协方差计算的中间结果
    glm::vec4 q = rot; // 旋转四元数 (w, x, y, z)，四元数是用于表示旋转的
    float r = q.x; // 四元数的实部 w
    float x = q.y; // 四元数的虚部 x
    float y = q.z; // 四元数的虚部 y
    float z = q.w; // 四元数的虚部 z

    // 步骤 2: 根据四元数计算旋转矩阵 R (3x3)
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    // 步骤 3: 构造缩放矩阵 S (3x3)，将缩放因子应用到对角线上
    glm::mat3 S = glm::mat3(1.0f); // 初始化为单位矩阵
    glm::vec3 s = mod * scale; // 对缩放因子乘上 mod 修改因子
    S[0][0] = s.x; // 设置 x 轴的缩放
    S[1][1] = s.y; // 设置 y 轴的缩放
    S[2][2] = s.z; // 设置 z 轴的缩放

    // 步骤 4: 将缩放矩阵和旋转矩阵相乘，得到矩阵 M
    glm::mat3 M = S * R;

    // 步骤 5: 从损失梯度 dL_dcov3Ds 中提取当前高斯的协方差矩阵梯度
    const float* dL_dcov3D = dL_dcov3Ds + 6 * idx; // 获取当前高斯的协方差梯度，idx 表示当前高斯的索引

    // 步骤 6: 将梯度分解为协方差矩阵的各个分量
    glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]); // 协方差矩阵的对角部分梯度
    glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]); // 协方差矩阵的非对角部分梯度

    // 步骤 7: 使用 per-element 协方差损失梯度构造协方差梯度矩阵 dL_dSigma
    glm::mat3 dL_dSigma = glm::mat3(
        dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
        0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
        0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
    );

    // 步骤 8: 计算关于矩阵 M 的损失梯度 dL_dM
    // dSigma_dM = 2 * M，因此乘以 2
    glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

    // 步骤 9: 计算关于 M 的转置矩阵 (dL_dMt) 的损失梯度
    glm::mat3 Rt = glm::transpose(R); // 计算旋转矩阵 R 的转置
    glm::mat3 dL_dMt = glm::transpose(dL_dM); // 计算 dL_dM 的转置

    // 步骤 10: 计算关于缩放因子 (scale) 的损失梯度
    glm::vec3* dL_dscale = dL_dscales + idx; // 获取存储当前高斯缩放梯度的位置
    dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]); // 对 x 轴的缩放梯度
    dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]); // 对 y 轴的缩放梯度
    dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]); // 对 z 轴的缩放梯度

    // 步骤 11: 将梯度矩阵 dL_dMt 按照缩放因子 (s.x, s.y, s.z) 调整
    dL_dMt[0] *= s.x;
    dL_dMt[1] *= s.y;
    dL_dMt[2] *= s.z;

    // 步骤 12: 计算关于四元数的损失梯度 (dL_dq)，四元数的梯度由旋转矩阵 M 的转置计算
    glm::vec4 dL_dq;
    dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
    dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
    dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
    dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

    // 步骤 13: 计算关于四元数的未归一化版本 (dL_drot) 的损失梯度
    float4* dL_drot = (float4*)(dL_drots + idx); // 获取存储当前高斯旋转梯度的位置
    *dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w }; // 存储四元数梯度
}


// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
// 这个核函数计算预处理步骤中的反向传播，除了协方差的计算和反转（这些由先前的内核调用处理）
template<int C>
__global__ void preprocessCUDA(
    int P, int D, int M,
    const float3* means,                // 输入的3D均值
    const int* radii,                   // 输入的半径
    const float* shs,                   // 输入的球谐函数系数
    const bool* clamped,                // 是否被夹紧的标志
    const glm::vec3* scales,            // 输入的缩放因子
    const glm::vec4* rotations,         // 输入的旋转四元数
    const float scale_modifier,         // 缩放因子的修正值
    const float* proj,                  // 投影矩阵（用于从3D投影到2D）
    const glm::vec3* campos,            // 相机位置（用于视图变换）
    const float3* dL_dmean2D,           // 2D均值的损失梯度
    glm::vec3* dL_dmeans,               // 输出的3D均值的损失梯度
    float* dL_dcolor,                   // 输出的颜色的损失梯度
    float* dL_dcov3D,                   // 输出的3D协方差矩阵的损失梯度
    float* dL_dsh,                      // 输出的球谐系数的损失梯度
    glm::vec3* dL_dscale,               // 输出的缩放因子的损失梯度
    glm::vec4* dL_drot,                 // 输出的旋转四元数的损失梯度
    float* dL_dopacity)                 // 输出的透明度的损失梯度
{
    // 获取当前线程的索引
    auto idx = cg::this_grid().thread_rank();

    // 如果索引超出范围或者半径无效，则跳过
    if (idx >= P || !(radii[idx] > 0))
        return;

    // 获取当前索引的3D均值
    float3 m = means[idx];

    // 步骤 1: 计算来自屏幕空间点的梯度
    // 将3D点变换为齐次坐标（投影到2D）
    float4 m_hom = transformPoint4x4(m, proj);
    float m_w = 1.0f / (m_hom.w + 0.0000001f); // 避免除以0的情况

    // 步骤 2: 根据2D均值的梯度，计算损失函数相对于3D均值的梯度
    glm::vec3 dL_dmean;
    // 计算投影矩阵对3D点的影响
    float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
    float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;

    // 计算损失梯度相对于3D均值的变化
    dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
    dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
    dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

    // 步骤 3: 更新dL_dmeans，即3D均值的损失梯度
    dL_dmeans[idx] += dL_dmean;

    // 步骤 4: 计算从SH系数到颜色的梯度更新（如果有SH系数的话）
    if (shs)
        computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

    // 步骤 5: 计算由缩放/旋转计算的协方差矩阵的梯度更新
    if (scales)
        computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}


// 渲染过程的反向传播版本
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,          // 每个线程块的像素范围
    const uint32_t* __restrict__ point_list,   // 点列表
    int W, int H,                              // 图像宽度和高度
    const float* __restrict__ bg_color,        // 背景颜色
    const float2* __restrict__ points_xy_image,// 每个点的2D坐标（屏幕空间）
    const float4* __restrict__ conic_opacity,  // 每个点的锥形不透明度
    const float* __restrict__ colors,          // 点的颜色
    const float* __restrict__ depths,          // 点的深度
    const float* __restrict__ final_Ts,        // 每个像素的最终透明度
    const uint32_t* __restrict__ n_contrib,    // 每个像素的贡献高斯数量
    const float* __restrict__ dL_dpixels,      // 像素的损失梯度
    const float* __restrict__ dL_invdepths,    // 逆深度的损失梯度
    float3* __restrict__ dL_dmean2D,           // 2D均值的损失梯度
    float4* __restrict__ dL_dconic2D,          // 2D锥形矩阵的损失梯度
    float* __restrict__ dL_dopacity,           // 不透明度的损失梯度
    float* __restrict__ dL_dcolors,            // 颜色的损失梯度
    float* __restrict__ dL_dinvdepths         // 逆深度的损失梯度
)
{
    // 计算每个线程块的范围信息
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };

    // 检查当前线程是否在图像范围内
    const bool inside = pix.x < W && pix.y < H;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

    // 设置迭代次数
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bool done = !inside;
    int toDo = range.y - range.x;

    // 定义共享内存，用于存储每个线程块的数据
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float collected_colors[C * BLOCK_SIZE];
    __shared__ float collected_depths[BLOCK_SIZE];

    // 计算当前像素的最终透明度
    const float T_final = inside ? final_Ts[pix_id] : 0;
    float T = T_final;

    // 获取最后贡献的高斯ID
    uint32_t contributor = toDo;
    const int last_contributor = inside ? n_contrib[pix_id] : 0;

    // 初始化梯度累加器
    float accum_rec[C] = { 0 };
    float dL_dpixel[C];
    float dL_invdepth;
    float accum_invdepth_rec = 0;
    if (inside)
    {
        // 从前向计算中获取像素和逆深度的梯度
        for (int i = 0; i < C; i++)
            dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
        if (dL_invdepths)
            dL_invdepth = dL_invdepths[pix_id];
    }

    // 存储上一个高斯的 alpha 值、颜色和逆深度
    float last_alpha = 0;
    float last_color[C] = { 0 };
    float last_invdepth = 0;

    // 计算像素坐标相对于归一化屏幕空间坐标的梯度
    const float ddelx_dx = 0.5 * W;
    const float ddely_dy = 0.5 * H;

    // 遍历所有高斯
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // 加载辅助数据到共享内存，倒序加载
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            const int coll_id = point_list[range.y - progress - 1];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            for (int i = 0; i < C; i++)
                collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];

            if (dL_invdepths)
                collected_depths[block.thread_rank()] = depths[coll_id];
        }
        block.sync();

        // 遍历高斯进行反向传播
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            // 更新当前高斯的ID，跳过已完成的高斯
            contributor--;
            if (contributor >= last_contributor)
                continue;

            // 计算混合值
            const float2 xy = collected_xy[j];
            const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            const float4 con_o = collected_conic_opacity[j];
            const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f)
                continue;

            const float G = exp(power);
            const float alpha = min(0.99f, con_o.w * G);
            if (alpha < 1.0f / 255.0f)
                continue;

            T = T / (1.f - alpha);
            const float dchannel_dcolor = alpha * T;

            // 传播梯度到每个高斯的颜色，并计算 alpha 的梯度
            float dL_dalpha = 0.0f;
            const int global_id = collected_id[j];
            for (int ch = 0; ch < C; ch++)
            {
                const float c = collected_colors[ch * BLOCK_SIZE + j];
                // 更新上一个颜色（用于下一次迭代）
                accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
                last_color[ch] = c;

                const float dL_dchannel = dL_dpixel[ch];
                dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
                // 更新颜色的梯度（原子操作，因为一个像素可能受到多个高斯的影响）
                atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
            }

            // 传播梯度到逆深度
            if (dL_dinvdepths)
            {
                const float invd = 1.f / collected_depths[j];
                accum_invdepth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_invdepth_rec;
                last_invdepth = invd;
                dL_dalpha += (invd - accum_invdepth_rec) * dL_invdepth;
                atomicAdd(&(dL_dinvdepths[global_id]), dchannel_dcolor * dL_invdepth);
            }

            dL_dalpha *= T;
            // 更新 alpha（用于下一次迭代）
            last_alpha = alpha;

            // 计算 alpha 对背景颜色的影响
            float bg_dot_dpixel = 0;
            for (int i = 0; i < C; i++)
                bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
            dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

            // 计算梯度，更新 2D 均值、协方差和不透明度的梯度
            const float dL_dG = con_o.w * dL_dalpha;
            const float gdx = G * d.x;
            const float gdy = G * d.y;
            const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
            const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

            // 更新 2D 均值的梯度
            atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
            atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

            // 更新 2D 协方差的梯度（对称矩阵）
            atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
            atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
            atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

            // 更新不透明度的梯度
            atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
        }
    }
}

void BACKWARD::preprocess(
    int P, int D, int M,                      // P: 点的数量，D: 高斯数目，M: 渲染参数
    const float3* means3D,                    // 3D 均值（每个点的 3D 均值）
    const int* radii,                         // 每个点的半径
    const float* shs,                         // Spherical Harmonics 系数（SH系数）
    const bool* clamped,                      // 是否固定（是否将高斯分布的点固定在某些位置）
    const float* opacities,                   // 每个点的透明度
    const glm::vec3* scales,                  // 每个点的缩放因子
    const glm::vec4* rotations,               // 每个点的旋转矩阵（四元数）
    const float scale_modifier,               // 缩放修正系数
    const float* cov3Ds,                      // 3D 协方差矩阵
    const float* viewmatrix,                  // 视图矩阵
    const float* projmatrix,                  // 投影矩阵
    const float focal_x, float focal_y,       // 焦距（x 和 y 方向的焦距）
    const float tan_fovx, float tan_fovy,     // 水平和垂直视场的切线值
    const glm::vec3* campos,                  // 相机位置
    const float3* dL_dmean2D,                 // 2D 均值的梯度（传递给 GPU 的梯度）
    const float* dL_dconic,                   // 2D 锥形矩阵的梯度
    const float* dL_dinvdepth,                // 逆深度的梯度
    float* dL_dopacity,                       // 不透明度的梯度
    glm::vec3* dL_dmean3D,                    // 3D 均值的梯度
    float* dL_dcolor,                         // 颜色的梯度
    float* dL_dcov3D,                        // 3D 协方差矩阵的梯度
    float* dL_dsh,                            // SH 系数的梯度
    glm::vec3* dL_dscale,                     // 缩放因子的梯度
    glm::vec4* dL_drot,                       // 旋转矩阵的梯度（四元数）
    bool antialiasing                         // 是否启用抗锯齿（抗锯齿用于减少图像的阶梯效应）
)
{
    // 调用计算 2D 锥形矩阵梯度的 CUDA 核函数
    computeCov2DCUDA<<<(P + 255) / 256, 256>>>(
        P, means3D, radii, cov3Ds, focal_x, focal_y,
        tan_fovx, tan_fovy, viewmatrix, opacities, 
        dL_dconic, dL_dopacity, dL_dinvdepth,
        (float3*)dL_dmean3D, dL_dcov3D, antialiasing);

    // 调用计算 3D 均值、协方差、颜色等梯度的 CUDA 核函数
    preprocessCUDA<NUM_CHANNELS><<< (P + 255) / 256, 256 >>>(
        P, D, M, (float3*)means3D, radii, shs, clamped,
        (glm::vec3*)scales, (glm::vec4*)rotations, scale_modifier,
        projmatrix, campos, (float3*)dL_dmean2D, 
        (glm::vec3*)dL_dmean3D, dL_dcolor, dL_dcov3D, 
        dL_dsh, dL_dscale, dL_drot, dL_dopacity);
}
