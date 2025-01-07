import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    """
    表示一个场景，包含相机数据、点云和高斯模型。该类负责加载和保存场景数据，并处理训练和测试相机。
    """
    
    gaussians : GaussianModel  # 高斯模型

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        初始化 Scene 对象，加载场景数据，包括高斯模型和相机数据。
        
        :param args: 模型参数
        :param gaussians: 高斯模型实例
        :param load_iteration: 要加载的训练迭代号，默认加载最后一个迭代
        :param shuffle: 是否打乱训练和测试相机数据，默认打乱
        :param resolution_scales: 分辨率比例列表，用于不同分辨率下的相机加载
        """
        self.model_path = args.model_path  # 模型路径
        self.loaded_iter = None  # 当前加载的迭代
        self.gaussians = gaussians  # 高斯模型

        # 如果提供了加载的迭代号，则加载该迭代的模型
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}  # 存储训练集相机数据
        self.test_cameras = {}   # 存储测试集相机数据

        # 根据输入路径的不同加载不同类型的场景
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 如果没有加载指定的迭代号，拷贝点云数据并创建相机数据
        if not self.loaded_iter:
            # 将点云数据拷贝到指定目录
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            # 将训练和测试相机数据合并
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))  # 将相机数据转为 JSON 格式
            # 保存相机数据为 JSON 文件
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 如果需要，打乱训练和测试相机数据
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # 对训练相机数据打乱
            random.shuffle(scene_info.test_cameras)  # 对测试相机数据打乱

        # 获取相机的最大视野半径
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 根据分辨率比例加载相机数据
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        # 如果指定了加载迭代，则加载对应的点云数据
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            # 否则创建新的点云数据
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        """
        保存当前的点云数据和曝光信息。

        :param iteration: 当前迭代
        """
        # 保存点云数据
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # 保存曝光信息
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }
        # 将曝光信息保存为 JSON 文件
        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        """
        获取指定分辨率比例下的训练集相机数据。

        :param scale: 分辨率比例，默认为 1.0
        :return: 训练集相机数据
        """
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """
        获取指定分辨率比例下的测试集相机数据。

        :param scale: 分辨率比例，默认为 1.0
        :return: 测试集相机数据
        """
        return self.test_cameras[scale]
