# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time Stereo Visual Odometry system with local non-linear least squares optimization. Estimates 6-DOF camera poses from stereo image sequences. Two pipeline modes are available:

1. **Standard Mode** (`main.py` + `StereoVO`): Traditional stereo matching → triangulation → pose estimation
2. **Depth Mode** (`main_depth.py` + `StereoVOWithDepth`): Uses pre-computed depth maps, skipping stereo matching and triangulation

Key modifications from original Stereo-Visual-SLAM-Odometry:
- Feature matching replaces optical flow (better for large inter-frame motion)
- ArUco marker detection as alternative to SIFT
- Depth-based pipeline for pre-computed stereo depth maps
- Point cloud concatenation using estimated poses

## Commands

```bash
# Install
conda env create -f setup/environment.yml
pip install -e .

# Standard VO pipeline (stereo matching + triangulation)
python main.py --config_path configs/params.yaml

# Depth-based VO pipeline (uses depth_meter.npy, skips triangulation)
python main_depth.py --config_path configs/params.yaml

# With point cloud concatenation
python main_depth.py --config_path configs/params.yaml --concatenate-pcd --pcd-voxel-size 0.001

# Disable Open3D visualization windows
python main_depth.py --config_path configs/params.yaml --no-vis

# Custom output path for concatenated point cloud
python main_depth.py --config_path configs/params.yaml --concatenate-pcd --pcd-output output.ply
```

## Architecture

### Two Pipeline Modes

**Standard Mode** (requires stereo image pairs):
```
Stereo Images → Feature Detection (both L/R) → Stereo Matching → Epipolar Filtering
    → Triangulation (DLT) → Frame Tracking → P3P + RANSAC → Pose Accumulation
```

**Depth Mode** (requires pre-computed depth maps):
```
Left Image + Depth Map → Feature Detection (L only) → Depth Lookup (get_3d_from_depth)
    → Frame Tracking → P3P + RANSAC → Pose Accumulation
```

The depth mode skips stereo matching and triangulation entirely by reading 3D coordinates directly from `depth_meter.npy`.

### Key Modules (`stereoVO/`)

| Module | Class/Function | Purpose |
|--------|----------------|---------|
| `model/stereoVO.py` | `StereoVO` | Standard pipeline with stereo matching + triangulation |
| `model/stereoVO_depth.py` | `StereoVOWithDepth` | Depth-based pipeline, calls `_update_stereo_state_with_depth()` |
| `model/drivers.py` | `StereoDrivers` | Base class with `_solve_pnp()`, `_do_optimization()` |
| `geometry/features.py` | `DetectionEngine` | SIFT detection + FLANN stereo matching |
| `geometry/features_aruco.py` | `ArucoDetectionEngine` | ArUco marker corner detection |
| `geometry/tracking_by_matching.py` | `MatchingEngine` | Frame-to-frame feature tracking via FLANN + F-matrix RANSAC |
| `geometry/epipolar.py` | `triangulate_points()` | DLT triangulation with reprojection error filtering |
| `geometry/depth_utils.py` | `get_3d_from_depth()` | Back-project 2D keypoints to 3D using depth map |
| `structures/state_machine.py` | `VO_StateMachine` | Per-frame state: features, descriptors, 3D points, pose |
| `datasets/CRC_Dataset.py` | `CRCDataset` | Load stereo images from cam0/cam1 folders |
| `datasets/Project_Dataset.py` | `ProjectDataset` | Load images + depth maps; auto-loads K.txt for intrinsics/baseline |

### Pose Representation

- `T_mat` (4x4): Camera pose in world frame. First frame is identity.
- `relative_pose` (4x4): Relative transformation from previous frame
- Accumulation: `T_mat[n] = T_mat[n-1] @ inv(relative_pose[n])`

### Point Cloud Concatenation

Each frame's point cloud is transformed to world coordinates:
```
P_world = T_mat @ [P_camera; 1]
```
Combined cloud is voxel-downsampled using Open3D.

## Dataset Formats

### CRC Dataset
```
dataset_path/
├── cam0/           # Left images (sorted by filename)
│   ├── 001.jpg
│   └── ...
└── cam1/           # Right images
    ├── 001.jpg
    └── ...
```
Camera intrinsics/extrinsics must be set in `params.yaml`.

### Project Dataset (with depth)
```
project_path/
└── {timestamp}/                  # Folder name used as frame ID
    ├── rect_left.jpg             # Rectified left image
    ├── rect_right.jpg            # Rectified right image (optional in depth mode)
    ├── depth_meter.npy           # (H, W) float32, depth in meters
    ├── cloud.ply                 # Point cloud for concatenation
    └── K.txt                     # Camera parameters:
                                  #   Line 1: fx 0 cx 0 fy cy 0 0 1 (9 values)
                                  #   Line 2: baseline in meters
```
`ProjectDataset` automatically loads intrinsics and baseline from `K.txt`.

## Configuration (`configs/params.yaml`)

```yaml
dataset:
    name: 'Project'      # 'CRC' (standard) or 'Project' (depth mode)
    path: './project'

geometry:
    depth_min: 0.5       # Min valid depth (meters) - depth mode only
    depth_max: 30.0      # Max valid depth (meters) - depth mode only

    detection:
        method: 'SIFT'   # 'SIFT' or 'ARUCO'

    pnpSolver:
        deltaT: 1.2      # Max translation magnitude per frame
        minRatio: 0.5    # Min inlier ratio for valid pose

    lsqsolver:
        enable: False    # Non-linear optimization (slower but more accurate)

debug:
    my_draw_matches: True   # Visualize feature matches (blocks execution)
```

## Key Implementation Details

### Feature Tracking (`MatchingEngine`)
- Uses FLANN with ratio test (default 0.5) for descriptor matching
- Validates matches using Fundamental Matrix + RANSAC
- Requires minimum 10 matches (`MIN_MATCH_COUNT`)
- If insufficient matches, previous pose is retained

### Depth-based 3D Point Extraction
`get_3d_from_depth()` in `depth_utils.py`:
```python
x = (u - cx) * z / fx
y = (v - cy) * z / fy
# Returns (N, 3) points and validity mask
```
Points filtered by: depth range, max radius from camera center.

### PnP Solver
- Uses `cv2.solvePnPRansac()` with P3P method
- Validates: translation < `deltaT`, inlier ratio > `minRatio`
- Falls back to previous pose if validation fails

## Output Files

```
output.path/                      # From params.yaml
├── svo_frames_poses.pkl          # Dict: {frame_name: 4x4 pose matrix}
├── svo_frames_poses.txt          # Human-readable poses
└── combined_pointcloud.ply       # If --concatenate-pcd used
```
