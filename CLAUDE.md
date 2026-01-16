# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stereo Visual Odometry system using pre-computed depth maps. Estimates 6-DOF camera poses from stereo image sequences by:
1. Detecting features in left images (SIFT or ArUco)
2. Getting 3D coordinates directly from depth maps (skips stereo matching & triangulation)
3. Tracking features between consecutive frames
4. Estimating pose using P3P + RANSAC
5. Accumulating poses to build camera trajectory

Includes point cloud concatenation using estimated poses.

## Commands

```bash
# Install
conda env create -f setup/environment.yml
pip install -e .

# Run VO pipeline
python main_depth.py --config_path configs/params_entry.yaml

# With point cloud concatenation
python main_depth.py --config_path configs/params_entry.yaml --concatenate-pcd --pcd-voxel-size 0.001

# Disable Open3D visualization windows
python main_depth.py --config_path configs/params_entry.yaml --no-vis

# Custom output path for concatenated point cloud
python main_depth.py --config_path configs/params_entry.yaml --concatenate-pcd --pcd-output output.ply
```

## Architecture

### Pipeline Flow

```
Left Image + Depth Map → Feature Detection (SIFT/ArUco) → Depth Lookup (get_3d_from_depth)
    → Frame-to-Frame Tracking (FLANN + F-matrix RANSAC) → P3P + RANSAC → Pose Accumulation
```

### Key Modules (`stereoVO/`)

| Module | Class/Function | Purpose |
|--------|----------------|---------|
| `model/stereoVO_depth.py` | `StereoVOWithDepth` | Main VO class with depth-based 3D extraction |
| `geometry/features.py` | `DetectionEngine` | SIFT detection + FLANN stereo matching |
| `geometry/features_aruco.py` | `ArucoDetectionEngine` | ArUco marker corner detection |
| `geometry/tracking_by_matching.py` | `MatchingEngine` | Frame-to-frame feature tracking |
| `geometry/tracking_aruco.py` | `ArucoMatchingEngine` | ArUco-based tracking |
| `geometry/depth_utils.py` | `get_3d_from_depth()` | Back-project 2D keypoints to 3D using depth map |
| `geometry/epipolar.py` | `triangulate_points()` | DLT triangulation (used in stereo matching) |
| `structures/state_machine.py` | `VO_StateMachine` | Per-frame state: features, descriptors, 3D points, pose |
| `datasets/Project_Dataset.py` | `ProjectDataset` | Load images + depth maps; auto-loads K.txt |
| `optimization/minimization.py` | `get_minimization()` | Reprojection error for LSQ optimization |

### Pose Representation

- `T_mat` (4x4): Camera pose in world frame. First frame is identity.
- `relative_pose` (4x4): Relative transformation from previous frame
- Accumulation: `T_mat[n] = T_mat[n-1] @ inv(relative_pose[n])`

### Point Cloud Concatenation

Each frame's point cloud is transformed to world coordinates:
```
P_world = T_mat @ [P_camera; 1]
```

## Dataset Format

```
project_path/
└── {timestamp}/                  # Folder name used as frame ID
    ├── rect_left.jpg             # Rectified left image
    ├── rect_right.jpg            # Rectified right image
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
    name: 'Project'
    path: './project'

geometry:
    depth_min: 0.5       # Min valid depth (meters)
    depth_max: 30.0      # Max valid depth (meters)

    detection:
        method: 'SIFT'   # 'SIFT' or 'ARUCO'

    pnpSolver:
        deltaT: 2.0      # Max translation magnitude per frame
        minRatio: 0.3    # Min inlier ratio for valid pose

    lsqsolver:
        enable: False    # Non-linear optimization (slower but more accurate)

debug:
    my_draw_matches: False   # Visualize feature matches (blocks execution)
```

## Key Implementation Details

### Feature Tracking (`MatchingEngine`)
- Uses FLANN with ratio test (default 0.5) for descriptor matching
- Validates matches using Fundamental Matrix + RANSAC
- Requires minimum 10 matches (`MIN_MATCH_COUNT`)
- If insufficient matches, previous pose is retained and frame marked as invalid

### Depth-based 3D Point Extraction
`get_3d_from_depth()` in `depth_utils.py`:
```python
x = (u - cx) * z / fx
y = (v - cy) * z / fy
# Returns (N, 3) points and validity mask
```

### PnP Solver
- Uses `cv2.solvePnPRansac()` with P3P method
- Validates: translation < `deltaT`, inlier ratio > `minRatio`
- Returns `pose_valid` flag indicating success/failure

## Output Files

```
output.path/                      # From params.yaml
├── svo_frames_poses.pkl          # Dict: {frame_name: 4x4 pose matrix}
├── svo_frames_poses.txt          # Human-readable poses
└── combined_pointcloud.ply       # If --concatenate-pcd used
```

## Deleted Files (no longer used)

- `main.py` - Replaced by `main_depth.py`
- `stereoVO/model/stereoVO.py` - Standard mode removed
- `stereoVO/model/drivers.py` - Merged into `stereoVO_depth.py`
- `stereoVO/datasets/CRC_Dataset.py` - Only Project dataset supported
- `stereoVO/geometry/tracking.py` - Optical flow tracking removed
