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

import os  # 用于文件和路径操作
import torch  # PyTorch 深度学习库
from random import randint  # 用于生成随机数
from utils.loss_utils import l1_loss, ssim  # 导入L1损失和SSIM损失函数
from gaussian_renderer import render, network_gui  # 导入渲染函数和GUI工具
import sys  # 系统操作
from scene import Scene, GaussianModel  # 场景和高斯模型
from utils.general_utils import safe_state, get_expon_lr_func  # 一些常用工具函数
import uuid  # 生成唯一标识符
from tqdm import tqdm  # 进度条
from utils.image_utils import psnr  # PSNR图像质量评估
from argparse import ArgumentParser, Namespace  # 处理命令行参数
from arguments import ModelParams, PipelineParams, OptimizationParams  # 定义模型、管道和优化参数类

# 尝试导入TensorBoard相关模块
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 尝试导入fused_ssim模块，计算改进的SSIM损失
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # 检查是否可以使用 Sparse Adam 优化器。如果不可用且用户选择了 Sparse Adam，程序退出
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0  # 初始化从哪个迭代开始
    tb_writer = prepare_output_and_logger(dataset)  # 初始化 TensorBoard 日志记录器
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)  # 创建高斯模型
    scene = Scene(dataset, gaussians)  # 创建场景对象，包含数据集和高斯模型
    gaussians.training_setup(opt)  # 设置高斯模型的训练配置
    if checkpoint:  # 如果提供了检查点，则恢复模型参数
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 根据数据集的设置，决定背景颜色（白色或黑色）
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转换为张量并放到 GPU 上

    # 设置 CUDA 事件，用于测量每次迭代的时间
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 检查是否使用 Sparse Adam 优化器
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    # 创建一个指数衰减的深度 L1 权重函数
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()  # 获取训练用的相机列表
    viewpoint_indices = list(range(len(viewpoint_stack)))  # 创建一个相机索引列表
    ema_loss_for_log = 0.0  # 用于计算损失的指数滑动平均
    ema_Ll1depth_for_log = 0.0  # 用于计算深度损失的指数滑动平均

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")  # 创建一个进度条
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # 检查并处理网络 GUI 连接
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # 接收来自 GUI 的数据并进行渲染
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # 使用自定义相机进行渲染
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    # 将渲染图像转换为字节格式并发送回 GUI
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None  # 出现异常时断开连接

        iter_start.record()  # 记录当前迭代的开始时间

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代时，增加 SH (Spherical Harmonics) 的级别，直到达到最大级别
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 从可用的相机中随机选择一个
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()  # 如果没有可用的相机，则重新加载
            viewpoint_indices = list(range(len(viewpoint_stack)))  # 重新创建相机索引列表
        rand_idx = randint(0, len(viewpoint_indices) - 1)  # 随机选择一个相机的索引
        viewpoint_cam = viewpoint_stack.pop(rand_idx)  # 从相机堆栈中取出相机
        vind = viewpoint_indices.pop(rand_idx)  # 从索引堆栈中移除已选择的索引

        # 如果当前迭代等于 debug_from，则启用调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 根据设置选择背景颜色：随机或固定
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 渲染当前相机视角的图像
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 如果相机包含 alpha mask，则进行处理
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()  # 将 alpha mask 移动到 GPU 上
            image *= alpha_mask  # 将 alpha mask 应用于渲染图像

        # 计算损失：L1 损失和 SSIM 损失
        gt_image = viewpoint_cam.original_image.cuda()  # 获取当前相机的原始图像作为地面真值
        Ll1 = l1_loss(image, gt_image)  # 计算 L1 损失
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))  # 如果可用，计算 Fused SSIM 损失
        else:
            ssim_value = ssim(image, gt_image)  # 否则使用标准 SSIM 损失

        # 总损失是 L1 损失和 SSIM 损失的加权和
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # 如果需要，添加深度正则化损失
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]  # 获取渲染的逆深度图
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()  # 获取相机的原始逆深度图
            depth_mask = viewpoint_cam.depth_mask.cuda()  # 获取深度掩码

            # 计算深度的 L1 损失
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure  # 深度损失的加权
            loss += Ll1depth  # 将深度损失添加到总损失中
            Ll1depth = Ll1depth.item()  # 转换为标量值
        else:
            Ll1depth = 0

        # 反向传播，计算梯度
        loss.backward()

        iter_end.record()  # 记录当前迭代的结束时间

        with torch.no_grad():
            # 更新指数滑动平均损失
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            # 每10次迭代更新一次进度条
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录训练报告，包括损失、时间等信息
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            # 每隔一段时间保存高斯模型
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 模型密度更新：用于增加或修剪高斯密度
            if iteration < opt.densify_until_iter:
                # 更新图像空间中的最大半径值
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 执行优化步骤，更新高斯模型的参数
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()  # 执行曝光优化器的步骤
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)  # 清空梯度
                if use_sparse_adam:
                    visible = radii > 0  # 仅对半径大于0的高斯体积进行优化
                    gaussians.optimizer.step(visible, radii.shape[0])  # 执行 Sparse Adam 优化步骤
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()  # 执行标准优化步骤
                    gaussians.optimizer.zero_grad(set_to_none=True)

            # 保存检查点（每隔一定步数）
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    # 如果未提供模型保存路径，自动生成一个唯一的路径
    if not args.model_path:
        # 尝试从环境变量获取 OAR 作业 ID（如果存在），用于生成唯一的文件夹名
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            # 如果没有 OAR 作业 ID，则生成一个 UUID 来作为文件夹的唯一标识
            unique_str = str(uuid.uuid4())

        # 模型路径设置为 "./output/" 加上 UUID 的前 10 个字符（确保路径唯一且简短）
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # 打印输出路径
    print("Output folder: {}".format(args.model_path))

    # 创建输出文件夹，如果文件夹已经存在，则不会抛出异常
    os.makedirs(args.model_path, exist_ok=True)

    # 将当前的配置参数写入输出文件夹中的 "cfg_args" 文件，方便后续查看和追踪
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        # 将传入的参数 (args) 转换为字典并保存到文件中
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建 TensorBoard 的日志记录器
    tb_writer = None
    if TENSORBOARD_FOUND:
        # 如果 TensorBoard 可用，则创建一个 TensorBoard 的 SummaryWriter
        tb_writer = SummaryWriter(args.model_path)
    else:
        # 如果 TensorBoard 不可用，则打印提示信息，不进行日志记录
        print("Tensorboard not available: not logging progress")

    # 返回 TensorBoard writer 或 None
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    # 如果 TensorBoard 写入器可用，则记录训练过程中的损失和迭代时间
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)  # 记录每个迭代的 L1 损失
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)  # 记录每个迭代的总损失
        tb_writer.add_scalar('iter_time', elapsed, iteration)  # 记录每个迭代的耗时

    # 如果当前迭代是测试迭代之一，则进行模型评估并保存测试结果
    if iteration in testing_iterations:
        # 清空显存缓存
        torch.cuda.empty_cache()

        # 设置测试和训练样本的配置，分别使用测试相机和一些训练相机
        validation_configs = ({
            'name': 'test',
            'cameras': scene.getTestCameras()  # 获取测试集的相机
        }, {
            'name': 'train',
            'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]  # 获取训练集的一部分相机
        })

        # 遍历所有验证配置，进行评估
        for config in validation_configs:
            # 如果相机配置非空，则进行评估
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0  # 初始化 L1 损失
                psnr_test = 0.0  # 初始化 PSNR 值
                # 对于每个视角，进行评估
                for idx, viewpoint in enumerate(config['cameras']):
                    # 渲染图像，并裁剪到 [0, 1] 范围内
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)  # 获取原始图像

                    # 如果进行的是训练-测试分离实验，则只保留图像的后半部分
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                    # 如果 TensorBoard 写入器可用，并且当前是前 5 个视角，则记录渲染图像和地面真值
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        # 仅在第一次测试时记录地面真值图像
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    # 计算 L1 损失和 PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                # 计算平均的 L1 损失和 PSNR 值
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                # 打印当前测试配置的评估结果
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                # 如果 TensorBoard 写入器可用，则记录测试集上的 L1 损失和 PSNR
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 如果 TensorBoard 写入器可用，则记录场景的透明度直方图和总点数
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        # 清空显存缓存
        torch.cuda.empty_cache()

if __name__ == "__main__":  # 确保当脚本被直接执行时才会执行以下代码
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")  # 创建命令行解析器并设置描述
    lp = ModelParams(parser)  # 配置模型参数解析
    op = OptimizationParams(parser)  # 配置优化参数解析
    pp = PipelineParams(parser)  # 配置数据处理管道参数解析
    parser.add_argument('--ip', type=str, default="127.0.0.1")  # 设置 IP 地址（默认为本地地址）
    parser.add_argument('--port', type=int, default=6009)  # 设置端口（默认为6009）
    parser.add_argument('--debug_from', type=int, default=-1)  # 设置调试从哪个迭代开始（默认为-1，表示不使用调试）
    parser.add_argument('--detect_anomaly', action='store_true', default=False)  # 是否启用异常检测
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])  # 设置进行测试的迭代次数
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])  # 设置保存模型的迭代次数
    parser.add_argument("--quiet", action="store_true")  # 是否以安静模式运行（不打印多余信息）
    parser.add_argument('--disable_viewer', action='store_true', default=False)  # 是否禁用视图渲染器
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 设置进行检查点保存的迭代次数
    parser.add_argument("--start_checkpoint", type=str, default = None)  # 设置从哪个检查点开始训练
    args = parser.parse_args(sys.argv[1:])  # 解析命令行参数
    args.save_iterations.append(args.iterations)  # 将总的训练迭代次数添加到保存迭代次数列表中

    print("Optimizing " + args.model_path)  # 打印模型路径，指示正在优化的模型

    # 初始化系统状态（如随机数生成器）
    safe_state(args.quiet)  # 调用 safe_state 函数进行系统初始化，如果 quiet 为 True，则减少输出信息

    # 启动 GUI 服务器，配置并开始训练
    if not args.disable_viewer:  # 如果没有禁用视图渲染器，则初始化 GUI 服务器
        network_gui.init(args.ip, args.port)  # 启动网络 GUI 服务器，传入 IP 和端口配置

    # 设置异常检测
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # 如果启用了异常检测，则设置 autograd 检测异常

    # 调用训练函数，传入配置好的参数
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # 训练完成后，输出提示信息
    print("\nTraining complete.")  # 输出训练完成的提示信息
