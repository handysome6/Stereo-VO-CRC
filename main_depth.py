"""
Main entry point for Stereo Visual Odometry with Pre-computed Depth Maps.

This script runs the VO pipeline using pre-computed depth maps from the
Project dataset structure, skipping the traditional stereo matching and
triangulation steps.

Usage:
    python main_depth.py --config_path configs/params.yaml
    python main_depth.py --config_path configs/params.yaml --no-vis
    python main_depth.py --config_path configs/params.yaml --concatenate-pcd
"""

import cv2
import argparse
import numpy as np
import open3d as o3d
import os
import pickle

from stereoVO.configs import yaml_parser
from stereoVO.datasets import ProjectDataset, ProjectFileNames, CRCDataset
from stereoVO.model.stereoVO_depth import StereoVOWithDepth
from stereoVO.model import StereoVO
from vis import draw_cam_poses


def parse_argument():
    parser = argparse.ArgumentParser(description='Stereo VO with Depth Maps')
    parser.add_argument('--config_path', default='configs/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='Disable visualization of camera trajectory')
    parser.add_argument('--save_poses', action='store_true', default=True,
                        help='Save poses to pickle file')
    parser.add_argument('--concatenate-pcd', action='store_true', default=False,
                        help='Concatenate point clouds using estimated poses')
    parser.add_argument('--pcd-voxel-size', type=float, default=0.01,
                        help='Voxel size for downsampling concatenated point cloud (meters)')
    parser.add_argument('--pcd-output', type=str, default=None,
                        help='Output path for concatenated point cloud')
    return parser.parse_args()


def concatenate_point_clouds(dataset, poses, frame_paths, voxel_size=0.01, valid_flags=None):
    """
    Concatenate point clouds from multiple frames using estimated camera poses.

    Each point cloud is transformed from camera coordinates to world coordinates
    using the camera pose (T_mat): P_world = T_mat @ P_camera

    Args:
        dataset: ProjectDataset instance
        poses: List of 4x4 camera pose matrices
        frame_paths: List of frame folder paths
        voxel_size: Voxel size for downsampling (meters), 0 to disable
        valid_flags: List of booleans indicating valid poses (skip invalid ones)

    Returns:
        combined_pcd: Combined Open3D point cloud in world coordinates
    """
    print("\n" + "=" * 60)
    print("Concatenating Point Clouds")
    print("=" * 60)

    combined_points = []
    combined_colors = []
    skipped_invalid = 0

    for i, (pose, frame_path) in enumerate(zip(poses, frame_paths)):
        # Skip frames with invalid poses
        if valid_flags is not None and not valid_flags[i]:
            print(f"  Frame {i} ({os.path.basename(frame_path)}): SKIPPED (invalid pose)")
            skipped_invalid += 1
            continue

        # Load point cloud from PLY file
        ply_path = os.path.join(frame_path, ProjectFileNames.CLOUD_PLY)

        if not os.path.exists(ply_path):
            print(f"  Frame {i}: PLY not found, skipping")
            continue

        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)

        if len(points) == 0:
            print(f"  Frame {i}: Empty point cloud, skipping")
            continue

        # Get colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            # Assign a unique color per frame for visualization
            colors = np.ones((len(points), 3)) * (i / len(poses))

        # Transform points from camera coordinates to world coordinates
        # P_world = T_mat @ P_camera
        # For homogeneous coordinates: [x, y, z, 1]^T
        points_homo = np.hstack([points, np.ones((len(points), 1))])  # (N, 4)
        points_world = (pose @ points_homo.T).T[:, :3]  # (N, 3)

        combined_points.append(points_world)
        combined_colors.append(colors)

        print(f"  Frame {i} ({os.path.basename(frame_path)}): {len(points)} points transformed")

    if skipped_invalid > 0:
        print(f"\n  Skipped {skipped_invalid} frames with invalid poses")

    if len(combined_points) == 0:
        print("No point clouds to concatenate!")
        return None

    # Combine all points
    all_points = np.vstack(combined_points)
    all_colors = np.vstack(combined_colors)

    print(f"\nTotal points before downsampling: {len(all_points)}")

    # Create combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(all_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Downsample using voxel grid
    if voxel_size > 0:
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
        print(f"Total points after downsampling (voxel={voxel_size}m): {len(combined_pcd.points)}")

    return combined_pcd


def main():
    args = parse_argument()

    # Load configuration
    params = yaml_parser(args.config_path)

    # Load camera parameters from config
    intrinsic_left = np.array(params.initial.intrinsic_left)
    intrinsic_right = np.array(params.initial.intrinsic_right)
    extrinsic = np.array(params.initial.extrinsic)

    # Initialize dataset based on type
    dataset_name = params.dataset.name
    dataset_path = params.dataset.path

    print(f"Loading dataset: {dataset_name} from {dataset_path}")

    if dataset_name == 'Project':
        # Use Project dataset with depth maps
        dataset = ProjectDataset(
            dataset_path,
            intrinsic_l=intrinsic_left,
            intrinsic_r=intrinsic_right,
            extrinsic=extrinsic
        )
        use_depth_mode = True
    elif dataset_name == 'CRC':
        # Use original CRC dataset
        dataset = CRCDataset(
            dataset_path,
            intrinsic_left,
            intrinsic_right,
            extrinsic
        )
        use_depth_mode = False
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")

    num_frames = len(dataset)
    print(f"Total frames: {num_frames}")

    # Get camera matrices
    cameraMatrix = dataset.intrinsic_l
    projectionMatrixL = dataset.PL
    projectionMatrixR = dataset.PR

    # Initialize the appropriate VO model
    if use_depth_mode:
        print("Using Depth Mode (StereoVOWithDepth)")
        model = StereoVOWithDepth(cameraMatrix, projectionMatrixL, projectionMatrixR, params)
    else:
        print("Using Standard Mode (StereoVO)")
        model = StereoVO(cameraMatrix, projectionMatrixL, projectionMatrixR, params)

    # Process frames
    poses = []
    pose_dict = {}
    frame_paths = []  # Store frame paths for point cloud concatenation
    pose_valid_flags = []  # Track which poses are valid (successfully estimated)

    for index in range(num_frames):
        print(f"\n{'='*60}")
        print(f"Processing frame {index + 1}/{num_frames}")
        print(f"{'='*60}")

        if use_depth_mode:
            # Load frame with depth map
            left_frame, right_frame, frame_path = dataset[index]
            depth_map = dataset.get_depth(index)

            # Get frame name for saving
            frame_name = os.path.basename(frame_path)
            frame_paths.append(frame_path)

            # Run VO with depth map
            pred_location, pred_orientation, pose, pose_valid = model(
                left_frame, right_frame, depth_map, index
            )
        else:
            # Original mode without depth
            left_frame, right_frame, left_img_path = dataset[index]
            frame_name = "A_" + os.path.split(left_img_path)[-1]
            frame_paths.append(os.path.dirname(left_img_path))

            pred_location, pred_orientation, pose = model(
                left_frame, right_frame, index
            )
            pose_valid = True  # Original model doesn't track validity

        poses.append(pose)
        pose_dict[frame_name] = pose
        pose_valid_flags.append(pose_valid)

        if pose_valid:
            print(f"Pose for {frame_name}: [VALID]")
        else:
            print(f"Pose for {frame_name}: [INVALID - kept previous, will skip in PCD concat]")
        print(pose)

    # Save poses
    if args.save_poses:
        saved_path = params.output.path
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        output_file = os.path.join(saved_path, "svo_frames_poses.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(pose_dict, f)
        print(f"\nPoses saved to: {output_file}")

        # Also save as readable text
        output_txt = os.path.join(saved_path, "svo_frames_poses.txt")
        with open(output_txt, "w") as f:
            for name, pose in pose_dict.items():
                f.write(f"{name}:\n")
                f.write(f"{pose}\n\n")
        print(f"Poses (text) saved to: {output_txt}")

    # Concatenate point clouds if requested
    if args.concatenate_pcd and use_depth_mode:
        combined_pcd = concatenate_point_clouds(
            dataset, poses, frame_paths,
            voxel_size=args.pcd_voxel_size,
            valid_flags=pose_valid_flags
        )

        if combined_pcd is not None:
            # Save combined point cloud
            saved_path = params.output.path
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)

            if args.pcd_output:
                pcd_output_path = args.pcd_output
            else:
                pcd_output_path = os.path.join(saved_path, "combined_pointcloud.ply")

            o3d.io.write_point_cloud(pcd_output_path, combined_pcd)
            print(f"\nCombined point cloud saved to: {pcd_output_path}")

            # Visualize combined point cloud
            if not args.no_vis:
                print("Visualizing combined point cloud...")
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                o3d.visualization.draw_geometries(
                    [combined_pcd, axis],
                    window_name="Combined Point Cloud",
                    width=1200,
                    height=800
                )

    # Visualize camera trajectory
    if not args.no_vis and len(poses) > 0:
        print("\nVisualizing camera trajectory...")
        cam_poses = draw_cam_poses(poses)
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries(
            [axis_pcd] + cam_poses,
            window_name="Visual Odometry - Camera Trajectory",
            width=800,
            height=600
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
