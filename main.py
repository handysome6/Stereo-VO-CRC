import cv2
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

from stereoVO.configs import yaml_parser
from stereoVO.datasets import CRCDataset
from stereoVO.model import StereoVO
from vis import draw_cam_poses
import os, pickle

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/params.yaml')
    return parser.parse_args()


def main():

    args = parse_argument()

    # Load the config file
    params = yaml_parser(args.config_path)

    intrinsic_left = np.array(params.initial.intrinsic_left)
    intrinsic_right = np.array(params.initial.intrinsic_right)
    extrinsic = np.array(params.initial.extrinsic)

    # Get data params using the dataloader
    dataset = CRCDataset(params.dataset.path, intrinsic_left, intrinsic_right, extrinsic)
    num_frames = len(dataset)
    cameraMatrix = dataset.intrinsic_l #[3,3]
    projectionMatrixL = dataset.PL #[3,4]
    projectionMatrixR = dataset.PR #[3,4]

    # Iniitialise the StereoVO model
    model = StereoVO(cameraMatrix, projectionMatrixL, projectionMatrixR, params)

    poses = []
    pose_dict = {}
    # Iterate over the frame and update the rotation and translation vector
    for index in range(num_frames):
        left_frame, right_frame, left_img_path = dataset[index]
        lef_raw_name = "A_" + os.path.split(left_img_path)[-1]

        # Do model prediction
        pred_location, pred_orientation, pose = model(left_frame, right_frame, index)
        # pose = np.eye(4)
        # pose[:3,:3] = pred_orientation
        # pose[:3,3] = pred_location
        poses.append(pose)
        pose_dict[lef_raw_name] = pose
        print("Pose: \n", pose)

    saved_path = params.output.path
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    with open(os.path.join(saved_path, "svo_frames_poses.pkl"), "wb") as f:
        pickle.dump(pose_dict, f)

    cam_poses = draw_cam_poses(poses)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([axis_pcd]+cam_poses,
                                        window_name="Visual Odometry",
                                        width=800,
                                        height=600)

if __name__ == "__main__":
    main()