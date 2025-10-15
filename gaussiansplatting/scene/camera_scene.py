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

import os

import numpy as np
from gaussiansplatting.scene.dataset_readers import sceneLoadTypeCallbacks
from gaussiansplatting.utils.camera_utils import cameraList_load


class CamScene:
    def __init__(self, source_path, h=512, w=512, aspect=-1, tag="train", transforms_path=""):
        """b
        :param path: Path to colmap scene main folder.
        """
        if aspect != -1:
            h = 512
            w = 512 * aspect

        if os.path.exists(os.path.join(source_path, "sparse")):
            if h == -1 or w == -1:
                scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, None, False)
                h = scene_info.train_cameras[0].height
                w = scene_info.train_cameras[0].width
                if w > 1920:
                    scale = w / 1920
                    h /= scale
                    w /= scale
            else:
                scene_info = sceneLoadTypeCallbacks["Colmap_hw"](source_path, h, w, None, False)
        elif os.path.exists(os.path.join(source_path, "transforms.json")):
            scene_info = sceneLoadTypeCallbacks["Nerfstudio_hw"](source_path, transforms_path, h, w, None, False)
        else:
            assert False, "Could not recognize scene type!"

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.scene_center = scene_info.nerf_normalization["center"]
        self.cameras_up = scene_info.nerf_normalization["up"]

        # if too much cameras, tends to cause CUDA OOM
        train_num, test_num = len(scene_info.train_cameras), len(scene_info.test_cameras)
        num_th = 50
        new_train_cameras, new_test_cameras = scene_info.train_cameras, scene_info.test_cameras
        if train_num>num_th or test_num>num_th:
            print(f"[INFO] Too many cameras, randomly select {num_th} cameras for training and testing.")
            train_num, test_num = min(train_num, num_th), min(test_num, num_th)
            train_ids = np.random.permutation(train_num)[:num_th]
            test_ids = np.random.permutation(test_num)[:num_th]
            new_train_cameras = [scene_info.train_cameras[i] for i in train_ids]
            new_test_cameras = [scene_info.test_cameras[i] for i in test_ids]
            scene_info = scene_info._replace(train_cameras=new_train_cameras, test_cameras=new_test_cameras)

        self.cameras = cameraList_load(scene_info.train_cameras if tag == "train" else scene_info.test_cameras, h, w)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
