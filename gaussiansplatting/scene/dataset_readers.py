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
import sys
from PIL import Image
from typing import NamedTuple
from gaussiansplatting.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from gaussiansplatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
from scipy.spatial.transform import Rotation

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    qvec: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers, cam_ups):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        cam_ups = np.hstack(cam_ups)
        avg_cam_up = np.mean(cam_ups, axis=1    )
        avg_cam_up /= np.linalg.norm(avg_cam_up)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal, avg_cam_up

    cam_centers = []
    cam_ups = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
        cam_ups.append(C2W[:3, 1:2])

    center, diagonal, up = get_center_and_diag(cam_centers, cam_ups)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius, "center": center, "up": up}

def readColmapCameras_hw(cam_extrinsics, cam_intrinsics, height, width, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        origin_height = intr.height
        origin_width = intr.width
        origin_aspect = origin_height/origin_width
        aspect = height/width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        qvec = np.array(extr.qvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            if origin_aspect > aspect: # shrink height
                FovY = focal2fov(focal_length_y, origin_width * aspect)
                FovX = focal2fov(focal_length_x, origin_width)
            else: # shrink width
                FovY = focal2fov(focal_length_y, origin_height)
                FovX = focal2fov(focal_length_x, origin_height/aspect)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length = intr.params[0]
            if origin_aspect > aspect: # shrink height
                FovY = focal2fov(focal_length, origin_width * aspect)
                FovX = focal2fov(focal_length, origin_width)
            else: # shrink width
                FovY = focal2fov(focal_length, origin_height)
                FovX = focal2fov(focal_length, origin_height/aspect)
        else:
            print(intr.model, "?????")
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, qvec=qvec,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        qvec = np.array(extr.qvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, qvec=qvec,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo_hw(path, h, w, images, eval, llffhold=8):
    sub_folder = "sparse/0" if os.path.exists(os.path.join(path, "sparse/0")) else "sparse"
        
    try:
        cameras_extrinsic_file = os.path.join(path, sub_folder, "images.bin")
        cameras_intrinsic_file = os.path.join(path, sub_folder, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, sub_folder, "images.txt")
        cameras_intrinsic_file = os.path.join(path, sub_folder, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    # with open("/mnt/16T/yejiangnan/work/instruct-gs2gs/data/in2n-data/face/transforms.json", "r") as fp:
    #     transforms_dict = json.load(fp)
    #     imgpath_matrix_list = []
    #     for frame in transforms_dict["frames"]:
    #         file_path = frame["file_path"]
    #         img_name = os.path.basename(file_path)
    #         imgpath_matrix_list.append((img_name, np.array(frame["transform_matrix"])))
    # imgpath_matrix_list.sort(key = lambda x : x[0])

    # R1 = np.array([i_m[1][:3, :3] for i_m in imgpath_matrix_list])
    # T1 = np.array([i_m[1][:3, 3] for i_m in imgpath_matrix_list])


    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras_hw(cam_extrinsics=cam_extrinsics, height=h, width=w, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # R2 = np.array([cam.R for cam in cam_infos])  # w2c -> c2w
    # T2 = np.array([cam.T for cam in cam_infos])

    # data = np.load("temp/transform_matrix_save.npz")
    # np.savez("temp/transform_matrix_save.npz", R_nerf=R1, T_nerf=T1, R_colmap=R2, T_colmap=T2)


   

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", True, extension=".jpg")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, sub_folder, "points3D.ply")
    bin_path = os.path.join(path, sub_folder, "points3D.bin")
    txt_path = os.path.join(path, sub_folder, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfo(path, images, eval, llffhold=8):
    sub_folder = "sparse/0" if os.path.exists(os.path.join(path, "sparse/0")) else "sparse"
        
    try:
        cameras_extrinsic_file = os.path.join(path, sub_folder, "images.bin")
        cameras_intrinsic_file = os.path.join(path, sub_folder, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, sub_folder, "images.txt")
        cameras_intrinsic_file = os.path.join(path, sub_folder, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)


    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # return List[CameraInfo]
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", True, extension=".jpg")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, sub_folder, "points3D.ply")
    bin_path = os.path.join(path, sub_folder, "points3D.bin")
    txt_path = os.path.join(path, sub_folder, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = matrix[:3, :3]
            q_vec = Rotation.from_matrix(R).as_quat()   # in w2c format
            R = R.T                                     # in c2w format
            T = matrix[:3, 3]                           # in w2c format

            # image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(cam_name)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=cam_name, image_name=image_name, width=image.size[0], height=image.size[1], qvec=q_vec))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfstudioInfo_hw(path, json_folder, h, w, images, eval, llffhold=8):
    print("Reading Training Transforms")
    with open(os.path.join(path, "transforms.json"), encoding="UTF-8") as file:
        meta = json.load(file)
    
    image_filenames = []
    c2ws = []

    fx_fixed = "fl_x" in meta
    fy_fixed = "fl_y" in meta
    cx_fixed = "cx" in meta
    cy_fixed = "cy" in meta
    height_fixed = "h" in meta
    width_fixed = "w" in meta

    fx = []
    fy = []
    cx = []
    cy = []
    height = []
    width = []

    # sort the frames by fname
    fnames = []
    for frame in meta["frames"]:
        filepath = Path(frame["file_path"])
        fnames.append(Path(path) / filepath)
    inds = np.argsort(fnames)
    frames = [meta["frames"][ind] for ind in inds]

    for frame in frames:
        filepath = Path(frame["file_path"])
        fname = Path(path) / filepath

        if not fx_fixed:
            assert "fl_x" in frame, "fx not specified in frame"
            fx.append(float(frame["fl_x"]))
        else:
            fx.append(float(meta["fl_x"]))
        if not fy_fixed:
            assert "fl_y" in frame, "fy not specified in frame"
            fy.append(float(frame["fl_y"]))
        else:
            fy.append(float(meta["fl_y"]))
        if not cx_fixed:
            assert "cx" in frame, "cx not specified in frame"
            cx.append(float(frame["cx"]))
        else:
            cx.append(float(meta["cx"]))
        if not cy_fixed:
            assert "cy" in frame, "cy not specified in frame"
            cy.append(float(frame["cy"]))
        else:
            cy.append(float(meta["cy"]))
        if not height_fixed:
            assert "h" in frame, "height not specified in frame"
            height.append(int(frame["h"]))
        else:
            height.append(int(meta["h"]))
        if not width_fixed:
            assert "w" in frame, "width not specified in frame"
            width.append(int(frame["w"]))
        else:
            width.append(int(meta["w"]))

        image_filenames.append(fname)
        
        pose = np.array(frame["transform_matrix"])
        pose[:, 1:3] *= -1  # convention opengl to opencv
        
        # assuming the scale factor can be read from nerfstudio outputs
        dataparser_transforms_path = Path(json_folder) / "dataparser_transforms.json"
        try:
            with open(dataparser_transforms_path, "r") as fp:
                trans_scale = json.load(fp)
                scale_factor = trans_scale["scale"]
        except:
            scale_factor = 1.0
        pose[3:, 3] /= scale_factor
        c2ws.append(pose)

    # construct cam_infos
    cam_infos = []
    for idx, c2w in enumerate(c2ws):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(c2ws)))
        sys.stdout.flush()

        origin_height = height[idx]
        origin_width = width[idx]
        origin_aspect = origin_height/origin_width
        aspect = h/w

        uid = 1
        R = c2w[:3, :3] # keep in c2w
        w2c = np.linalg.inv(c2w)
        T = w2c[:3, 3]  # keep in w2c
        qvec = rotmat2qvec(w2c[:3, :3])  # keep in w2c

        focal_length_x = fx[idx]
        focal_length_y = fy[idx]
        if origin_aspect > aspect: # shrink height
            FovY = focal2fov(focal_length_y, origin_width * aspect)
            FovX = focal2fov(focal_length_x, origin_width)
        else: # shrink width
            FovY = focal2fov(focal_length_y, origin_height)
            FovX = focal2fov(focal_length_x, origin_height/aspect)

        image_path = image_filenames[idx]
        image_name = image_path.name.split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, qvec=qvec,
                              image_path=image_path, image_name=image_name, width=w, height=h)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # we do not need a init point cloud for editing 
    pcd = None
    ply_path = ""

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromScannetppTransforms(path, transformsfile, depths_folder, white_background, eval, extension=".png", new_h=512, new_w=512):
    cam_train_infos, cam_test_infos = [], []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        # camera_model = contents["camera_model"]
        # orientation_override = contents["orientation_override"]
        frame_same = False
        if "fl_x" in contents:
            frame_same = True
            fl_x = contents["fl_x"]
            fl_y = contents["fl_y"]
            cx = contents["cx"]
            cy = contents["cy"]
            h = contents["h"]
            w = contents["w"]
        
        has_test_split = "test_frames" in contents
        LLFF_hold_num = 8

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            image_path = os.path.join(path, "resized_undistorted_images", frame["file_path"])
            if not frame_same:
                fl_x = frame["fl_x"]
                fl_y = frame["fl_y"]
                cx = frame["cx"]
                cy = frame["cy"]
                h = frame["h"]
                w = frame["w"]


            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            qvec = rotmat2qvec(w2c[:3, :3])  # keep in w2c

            image_name = Path(image_path).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            origin_height = h
            origin_width = w
            origin_aspect = origin_height/origin_width
            aspect = new_h/new_w
            focal_length_x = fl_x
            focal_length_y = fl_y
            if origin_aspect > aspect: # shrink height
                FovY = focal2fov(focal_length_y, origin_width * aspect)
                FovX = focal2fov(focal_length_x, origin_width)
            else: # shrink width
                FovY = focal2fov(focal_length_y, origin_height)
                FovX = focal2fov(focal_length_x, origin_height/aspect)

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            if eval and not has_test_split and (idx%LLFF_hold_num==0):
                 
                cam_test_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, qvec=qvec,
                              image_path=image_path, image_name=image_name, width=w, height=h))

            else:
                cam_train_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, qvec=qvec,
                              image_path=image_path, image_name=image_name, width=w, height=h))
                
        if has_test_split:
            test_frames = contents["test_frames"]
            for idx, frame in enumerate(test_frames):
                image_path = os.path.join(path, "resized_undistorted_images", frame["file_path"])
                if not frame_same:
                    fl_x = frame["fl_x"]
                    fl_y = frame["fl_y"]
                    cx = frame["cx"]
                    cy = frame["cy"]
                    h = frame["h"]
                    w = frame["w"]


                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]
                qvec = rotmat2qvec(w2c[:3, :3])  # keep in w2c

                image_name = Path(image_path).stem
                image = Image.open(image_path)

                im_data = np.array(image.convert("RGBA"))

                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

                origin_height = h
                origin_width = w
                origin_aspect = origin_height/origin_width
                aspect = new_h/new_w
                focal_length_x = fl_x
                focal_length_y = fl_y
                if origin_aspect > aspect: # shrink height
                    FovY = focal2fov(focal_length_y, origin_width * aspect)
                    FovX = focal2fov(focal_length_x, origin_width)
                else: # shrink width
                    FovY = focal2fov(focal_length_y, origin_height)
                    FovX = focal2fov(focal_length_x, origin_height/aspect)

                depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

                if eval:
                    cam_test_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, qvec=qvec,
                              image_path=image_path, image_name=image_name, width=w, height=h))
                else:
                    cam_train_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, qvec=qvec,
                              image_path=image_path, image_name=image_name, width=w, height=h))
    
    return cam_train_infos, cam_test_infos

def readScannetppInfo_hw(path, white_background, depths, eval, h, w):
    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading All Transforms")
    train_cam_infos, test_cam_infos = readCamerasFromScannetppTransforms(path, "nerfstudio/transforms_undistorted.json", depths_folder, white_background, eval, h, w)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "colmap", "points3D.ply")
    bin_path = os.path.join(path, "colmap", "points3D.bin")
    txt_path = os.path.join(path, "colmap", "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Colmap_hw": readColmapSceneInfo_hw,
    "Blender" : readNerfSyntheticInfo,
    "Nerfstudio_hw" : readNerfstudioInfo_hw,
    "ScanNetpp": readScannetppInfo_hw,
}
