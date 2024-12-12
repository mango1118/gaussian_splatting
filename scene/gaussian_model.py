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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        # 从缩放旋转因子里构建协方差矩阵
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # 旋转矩阵乘缩放矩阵，得到高斯椭球的变化，得到L矩阵
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # L乘L第一维和第二维的转置，第零维是高斯数量，所以跳过
            # 构建出真实的协方差矩阵
            actual_covariance = L @ L.transpose(1, 2)
            # 只保留上三角，因为是对称矩阵
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # 定义激活函数，缩放因子的激活函数就是exp
        self.scaling_activation = torch.exp
        # 对应缩放因子的反激活函数是log
        self.scaling_inverse_activation = torch.log
        # 协方差矩阵没用激活函数，因为旋转和缩放都激活过了，所以直接用刚才的方法构造
        self.covariance_activation = build_covariance_from_scaling_rotation
        # 不透明度的激活函数用sigmoid，为了不透明度在0-1之间
        self.opacity_activation = torch.sigmoid
        # 对应的反函数就是反激活函数
        self.inverse_opacity_activation = inverse_sigmoid
        # 旋转操作的激活函数是归一化函数
        self.rotation_activation = torch.nn.functional.normalize

    # 对变量进行初始化，设置成0或者空
    def __init__(self, sh_degree, optimizer_type="default"):
        # 球谐函数的阶数
        self.active_sh_degree = 0
        # 优化类型
        self.optimizer_type = optimizer_type
        # 球谐函数的最高阶数是传进来的
        self.max_sh_degree = sh_degree 
        # 椭球位置
        self._xyz = torch.empty(0)
        # 球谐函数的直流分量，是指球谐级数中最低阶的部分，即阶数l=0的分量。
        self._features_dc = torch.empty(0)
        # 球谐函数的高阶分量，高阶分量是指球谐级数中阶数l>0的部分。
        # 这些分量描述了球面数据的角度变化和细节。
        self._features_rest = torch.empty(0)
        # 缩放因子
        self._scaling = torch.empty(0)
        # 旋转因子
        self._rotation = torch.empty(0)
        # 不透明度
        self._opacity = torch.empty(0)
        # 投影到平面后的二维高斯分布的最大半径
        self.max_radii2D = torch.empty(0)
        # 点位置的梯度累积值
        self.xyz_gradient_accum = torch.empty(0)
        # 统计的分母数量，梯度累积值需要除以分母数量来计算每个高斯分布的平均梯度
        self.denom = torch.empty(0)
        # 优化器
        self.optimizer = None
        # 百分比，做密度控制
        self.percent_dense = 0
        # 学习率因子
        self.spatial_lr_scale = 0
        # 创建激活函数的方法
        self.setup_functions()

    # 获得这个高斯点
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    # 恢复模型的状态和相关设置，model_args是包含多个值的元组，还有各种其他参数等
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # 获取变量时，返回的是激活后的变量，所以需要用反激活函数来把变量提取出来
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # 迭代球谐函数的阶数，如果球谐函数的阶数小于规定的最大阶数，运行这个方法之后阶数就会增加
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 从点云文件中创建数据
    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float): # 对象，传入点云，学习率的变化因子
        # 将变化因子传入对象的学习率
        self.spatial_lr_scale = spatial_lr_scale
        # 创建张量来保存点云数据，把数组类型的点云数据存到张量里
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # 把点云颜色存到RGB张量里面，转成球谐函数的系数
        # 这里只存了零阶的，也就是直流分量的球谐函数，其他后面再加上高阶的
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # 定义了张量，维度是高斯分布总数，3个通道，球谐函数的系数数量
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # 第0阶就是直流分量
        features[:, :3, 0 ] = fused_color
        # 高阶分量先定义为0，默认点云的点只有点的颜色
        features[:, 3:, 1:] = 0.0

        # 打印初始化的点的数量
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 计算distance，首先根据点云创建一个张量，设置最小距离0.0000001，distCUDA2函数在simple-knn里，计算对应点云最近邻居
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 高斯球的半径就是到最近的三个高斯点的距离的平均值，有最小距离因此不会有重合的高斯椭球
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 创建旋转变量，维度是N*4的张量
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # 存四位数w，x，y，z，将w设成1，则2arcosw就是0，其他xyz也是0，单位四元数的整体值就是0，即将旋转因子初始化为0
        rots[:, 0] = 1

        # 初始化不透明度
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 高斯椭球的分布位置用参数存储，规定梯度将来进行优化
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # 球谐函数的直流分量，规定梯度将来进行优化
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # 球谐函数的高阶分量，规定梯度将来进行优化
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # 旋转
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        # 缩放
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        # 不透明度
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # 二维投影分布的高斯最大半径
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        # 几种参数的传递
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 动态的学习率存储，每个参数的学习率都不一样
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # Adam优化器
        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    # 更新学习率
    def update_learning_rate(self, iteration):
        ''' 每一步迭代的学习率调度 '''

        # 如果没有使用预训练曝光值（即pretrained_exposures为None）
        if self.pretrained_exposures is None:
            # 对于曝光优化器（exposure_optimizer）中的每个参数组，更新其学习率
            for param_group in self.exposure_optimizer.param_groups:
                # 使用预定义的学习率调度函数（exposure_scheduler_args）来更新学习率
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        # 对于主优化器（optimizer）中的每个参数组，检查是否需要更新学习率
        for param_group in self.optimizer.param_groups:
            # 如果当前参数组的名称是 "xyz"，则更新该组的学习率
            if param_group["name"] == "xyz":
                # 使用 xyz 参数调度函数（xyz_scheduler_args）来计算该参数组的学习率
                lr = self.xyz_scheduler_args(iteration)
                # 将计算后的学习率赋值给该参数组
                param_group['lr'] = lr
                # 返回更新后的学习率
                return lr

    # 从参数里面创建列表
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # 结果保存到点云文件
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 重置不透明度
    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # 加载点云文件
    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # 将张量转到优化器里，用于优化
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # 参数发生大的变化，但优化器里存的动量应该是不变的，保证原有的状态不丢失，最后损失下降就是一个平滑的下降状态
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 重置优化器
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 要保留哪些状态
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 删除点，在做高斯点的修剪，不需要的高斯点就用mask删掉
    def prune_points(self, mask):
        # 选择要保留哪些点
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    # 将新的张量合并到优化器中，适用于扩展模型参数
    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将新的张量合并到优化器的参数中，并更新优化器的状态。
        参数:
            tensors_dict (dict): 包含扩展张量的字典，键是参数名称，值是对应的扩展张量。
        返回:
            optimizable_tensors (dict): 更新后的优化器可优化参数字典，键是参数名称，值是合并后的张量。
        """
        optimizable_tensors = {}  # 存储更新后的可优化张量

        # 遍历优化器中的所有参数组（param_groups）
        for group in self.optimizer.param_groups:
            # 确保每个参数组只有一个参数
            assert len(group["params"]) == 1
            # 获取当前参数组对应的扩展张量
            extension_tensor = tensors_dict[group["name"]]
            # 获取该参数在优化器中的存储状态（如果存在的话）
            stored_state = self.optimizer.state.get(group['params'][0], None)
            # 如果该参数已经存在优化器状态中（即该参数已经被优化过），则进行扩展
            if stored_state is not None:
                # 将“exp_avg”和“exp_avg_sq”扩展，补齐与新参数形状一致的零张量
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)
                # 删除原先的参数状态（已扩展过的参数将会重新计算）
                del self.optimizer.state[group['params'][0]]
                # 将原参数和扩展张量连接，创建新的可训练参数，并将其赋值为新的参数组
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                # 将扩展后的参数状态重新存入优化器的状态字典
                self.optimizer.state[group['params'][0]] = stored_state
                # 将更新后的参数添加到可优化参数字典中
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果优化器中没有该参数的状态，直接将原始参数与扩展张量连接
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                # 将新参数添加到可优化参数字典中
                optimizable_tensors[group["name"]] = group["params"][0]
        # 返回更新后的可优化参数字典
        return optimizable_tensors

    # 给自适应密度添加新的高斯点要用到的函数
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        # 一堆属性，要添加的属性赋值给这些属性，
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        # 调用创建优化器的方法，将这些属性添加到优化器里
        # 创建一个优化器张量
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # 把新的值赋给新的对象，创造新的高斯点
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # 自适应密度的分裂操作，参数有目前优化过程中的梯度、设定的梯度阈值、场景范围、特定常数2
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # 先读取高斯分布的总数
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # 创建一个全0的张量，大小就是点的高斯分布的总数的大小，用这个变量存储每个高斯分布现在的梯度
        padded_grad = torch.zeros((n_init_points), device="cuda")
        # 根据这些梯度扩展一个维度
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # 然后生成掩码，如果梯度大于给定梯度的阈值，就根据掩码做一个标记
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 再进行一次筛选，对于这个高斯分布的缩放因子中，最大的一个维度的值大于场景的范围乘以一个比例因子，（就是高斯分布的大小已经大于要求的场景范围）
        # 这个高斯就要进行分裂
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        # 新的高斯分布的标准差取自于原本高斯分布的标准差，然后把他扩展成两个
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        # 新高斯分布的位置规定到全0的位置
        means =torch.zeros((stds.size(0), 3),device="cuda")
        # 新高斯分布就通过原本的均值和标准差进行设计
        samples = torch.normal(mean=means, std=stds)
        # 创建旋转矩阵，也是根据需要分裂的高斯进行创建
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # 用创建好的高斯分布，先增加一个维度，然后缩放因子跟旋转矩阵相乘，得到协方差矩阵，然后再删掉新增的维度
        # 加上原本高斯分布的位置，就得到新高斯分布的位置
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 获取新的缩放因子之后，除以0.8*2，也就是原文中的1.6，两个变小成为小的高斯分布
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # 旋转矩阵，球谐函数，不透明度都调用原本的
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # 把新的变量添加到高斯分布里
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # 创建一个过滤器，把之前的一些高斯分布删掉
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # 清除掉中间变量
        self.prune_points(prune_filter)

    # 克隆
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 高斯分布的梯度的阈值大于设定的梯度阈值，就要进行克隆，判断形状是否小于场景设定的形状范围
        # 同样是一个张量
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 满足条件就要标记为需要克隆
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 原高斯分布的所有变量都添加到新变量里，增加一群新高斯（用tensor标记）
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    # 高斯椭球的剔除与密度调整
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        """
        对高斯椭球进行密度调整（densify）和剔除（prune）的操作。

        参数:
            max_grad (float): 最大允许的梯度，用于控制分裂和克隆。
            min_opacity (float): 最小不透明度阈值，低于该值的高斯椭球将被剔除。
            extent (float): 场景范围，通常用于归一化或大小判断。
            max_screen_size (float or None): 屏幕上允许的最大高斯椭球半径，超过该值的椭球将被剔除。
            radii (torch.Tensor): 当前高斯椭球的半径张量。

        主要步骤:
        1. 计算梯度，归零所有无效值（NaN）。
        2. 调用克隆和分裂函数，调整高斯椭球的密度。
        3. 根据不透明度、屏幕尺寸和场景范围剔除不符合要求的高斯椭球。
        4. 清理中间变量，释放显存。
        """
        # 1. 计算梯度平均值
        # 使用累计梯度值（xyz_gradient_accum）除以分母（denom）得到平均梯度。
        # grads 是一个一维张量，代表每个高斯椭球的梯度强度。
        grads = self.xyz_gradient_accum / self.denom

        # 处理无效值：将 grads 中的所有 NaN 值替换为 0。
        grads[grads.isnan()] = 0.0

        # 2. 调整高斯椭球的密度
        # 将当前高斯椭球半径临时存储，以便后续操作。
        self.tmp_radii = radii

        # 调用密度调整方法:
        # densify_and_clone: 根据梯度和场景范围克隆高斯椭球。
        self.densify_and_clone(grads, max_grad, extent)
        # densify_and_split: 根据梯度和场景范围分裂高斯椭球。
        self.densify_and_split(grads, max_grad, extent)

        # 3. 创建剔除掩码
        # 根据不透明度阈值剔除：创建布尔张量 prune_mask，
        # 标记所有不透明度低于 min_opacity 的高斯椭球。
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        # 如果设置了屏幕尺寸约束（max_screen_size 不为 None），则进行进一步剔除。
        if max_screen_size:
            # 检查二维高斯分布半径是否超过屏幕允许的最大尺寸。
            big_points_vs = self.max_radii2D > max_screen_size
            # 检查高斯分布的三维尺度是否超过场景范围的 10%。
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            # 合并剔除条件：如果某个高斯椭球满足任一条件，则标记为需要剔除。
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # 根据剔除掩码（prune_mask）执行剔除操作。
        self.prune_points(prune_mask)

        # 恢复临时存储的半径信息并清理。
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        # 4. 清理缓存
        # 调用 PyTorch 的显存清理函数，释放未使用的显存以优化内存使用。
        torch.cuda.empty_cache()

    # 添加自适应密度控制过程中的状态，就是记录需要累加的梯度
    '''
    整体流程总结：
    计算每个点的梯度大小：首先，通过 viewspace_point_tensor.grad[update_filter, :2] 获取需要更新的点在 x 和 y 方向上的梯度值。
    计算梯度的范数：然后，使用 torch.norm(..., dim=-1, keepdim=True) 计算这些梯度的 L2 范数，即每个点在 x 和 y 方向上的梯度强度。
    累积梯度信息：最后，将计算得到的梯度范数累加到 self.xyz_gradient_accum[update_filter] 中，以跟踪这些点的梯度变化。
    每处理一个点，就给分母+1,因为要计算平均梯度
    '''
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 累计梯度，把x和y方向上的梯度给标记起来，添加到对应的计数器
        # 计算并累积与视空间点相关的梯度信息（这里只关注前两个维度，即x和y方向的梯度）
        # `viewspace_point_tensor.grad` 代表该视空间点的梯度
        # `update_filter` 是一个布尔数组，用来筛选出需要更新的点
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # `torch.norm` 计算梯度的范数，dim=-1 表示计算每个点在最后一个维度的范数，keepdim=True 保持维度不变
        # 这里假设我们只关心 x 和 y 方向的梯度，因此对 `viewspace_point_tensor.grad[update_filter,:2]` 进行了切片
        # 累积计数，统计每个点更新的次数
        # `denom` 用于存储每个点的更新次数，`update_filter` 用来筛选出需要更新的点
        # 每为高斯点累计一次梯度，就给它的计数+1，因为后续要作为分母得到每个高斯点的平均梯度
        self.denom[update_filter] += 1

