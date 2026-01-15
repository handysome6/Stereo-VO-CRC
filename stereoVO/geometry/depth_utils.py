"""
Depth Utility Functions for Stereo VO Pipeline

Functions to convert 2D keypoints to 3D points using pre-computed depth maps,
replacing the traditional stereo matching and triangulation steps.
"""

import numpy as np

__all__ = ['get_3d_from_depth', 'filter_3d_points', 'get_3d_from_depth_with_filter']


def get_3d_from_depth(keypoints_2d, depth_map, K, interpolate=False):
    """
    Back-project 2D keypoints to 3D points using depth map.

    Args:
        keypoints_2d: (N, 2) array of pixel coordinates (u, v)
        depth_map: (H, W) depth map in meters
        K: (3, 3) camera intrinsic matrix
        interpolate: If True, use bilinear interpolation for sub-pixel accuracy

    Returns:
        pts3D: (N, 3) array of 3D points in camera coordinate frame
        valid_mask: (N,) boolean array indicating valid points
    """
    if len(keypoints_2d) == 0:
        return np.array([]).reshape(0, 3), np.array([], dtype=bool)

    keypoints_2d = np.asarray(keypoints_2d)
    if keypoints_2d.ndim == 1:
        keypoints_2d = keypoints_2d.reshape(1, 2)

    N = keypoints_2d.shape[0]
    H, W = depth_map.shape[:2]

    # Extract intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pts3D = np.zeros((N, 3), dtype=np.float64)
    valid_mask = np.zeros(N, dtype=bool)

    for i, (u, v) in enumerate(keypoints_2d):
        # Get integer pixel coordinates
        u_int, v_int = int(round(u)), int(round(v))

        # Boundary check
        if not (0 <= v_int < H and 0 <= u_int < W):
            continue

        # Get depth value
        if interpolate and 0 < u < W - 1 and 0 < v < H - 1:
            # Bilinear interpolation for sub-pixel accuracy
            z = _bilinear_interpolate(depth_map, u, v)
        else:
            z = depth_map[v_int, u_int]

        # Valid depth check
        if z > 0 and np.isfinite(z):
            # Back-project to 3D
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pts3D[i] = [x, y, z]
            valid_mask[i] = True

    return pts3D, valid_mask


def _bilinear_interpolate(depth_map, u, v):
    """Bilinear interpolation of depth value at sub-pixel location."""
    H, W = depth_map.shape[:2]

    u0, v0 = int(u), int(v)
    u1, v1 = min(u0 + 1, W - 1), min(v0 + 1, H - 1)

    # Get the four corner depths
    d00 = depth_map[v0, u0]
    d01 = depth_map[v0, u1]
    d10 = depth_map[v1, u0]
    d11 = depth_map[v1, u1]

    # Check if all corners have valid depth
    if d00 <= 0 or d01 <= 0 or d10 <= 0 or d11 <= 0:
        # Fall back to nearest neighbor
        return depth_map[int(round(v)), int(round(u))]

    # Bilinear interpolation weights
    wu = u - u0
    wv = v - v0

    # Interpolate
    d = (1 - wv) * ((1 - wu) * d00 + wu * d01) + wv * ((1 - wu) * d10 + wu * d11)
    return d


def filter_3d_points(pts3D, valid_mask,
                     min_depth=0.5, max_depth=30.0,
                     max_radius=50.0,
                     x_range=(-20, 20), y_range=(-10, 10)):
    """
    Filter 3D points based on depth and spatial constraints.

    Args:
        pts3D: (N, 3) array of 3D points
        valid_mask: (N,) boolean array of initially valid points
        min_depth: Minimum valid depth (meters)
        max_depth: Maximum valid depth (meters)
        max_radius: Maximum radial distance from camera center
        x_range: (min_x, max_x) valid X coordinate range
        y_range: (min_y, max_y) valid Y coordinate range

    Returns:
        filtered_pts3D: (M, 3) filtered 3D points
        filter_mask: (N,) boolean array indicating which original points passed filtering
    """
    N = len(pts3D)
    filter_mask = valid_mask.copy()

    for i in range(N):
        if not filter_mask[i]:
            continue

        x, y, z = pts3D[i]

        # Depth range check
        if z < min_depth or z > max_depth:
            filter_mask[i] = False
            continue

        # Spatial bounds check
        if x < x_range[0] or x > x_range[1]:
            filter_mask[i] = False
            continue

        if y < y_range[0] or y > y_range[1]:
            filter_mask[i] = False
            continue

        # Radial distance check
        radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if radius > max_radius:
            filter_mask[i] = False
            continue

    return pts3D[filter_mask], filter_mask


def get_3d_from_depth_with_filter(keypoints_2d, depth_map, K,
                                   min_depth=0.5, max_depth=30.0,
                                   max_radius=50.0,
                                   x_range=(-20, 20), y_range=(-10, 10),
                                   interpolate=False):
    """
    Combined function to get 3D points from depth and filter them.

    Args:
        keypoints_2d: (N, 2) array of pixel coordinates
        depth_map: (H, W) depth map in meters
        K: (3, 3) camera intrinsic matrix
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        max_radius: Maximum radial distance
        x_range: Valid X range
        y_range: Valid Y range
        interpolate: Use bilinear interpolation

    Returns:
        pts3D_filtered: (M, 3) filtered 3D points
        valid_indices: (M,) indices of valid points in original keypoints array
    """
    # Get 3D points from depth
    pts3D, valid_mask = get_3d_from_depth(keypoints_2d, depth_map, K, interpolate)

    # Filter 3D points
    _, filter_mask = filter_3d_points(
        pts3D, valid_mask,
        min_depth=min_depth, max_depth=max_depth,
        max_radius=max_radius,
        x_range=x_range, y_range=y_range
    )

    # Get indices of valid points
    valid_indices = np.where(filter_mask)[0]

    return pts3D[filter_mask], valid_indices


def reproject_3d_to_2d(pts3D, K):
    """
    Project 3D points back to 2D for verification.

    Args:
        pts3D: (N, 3) array of 3D points
        K: (3, 3) camera intrinsic matrix

    Returns:
        pts2D: (N, 2) array of 2D pixel coordinates
    """
    if len(pts3D) == 0:
        return np.array([]).reshape(0, 2)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pts2D = np.zeros((len(pts3D), 2))
    for i, (x, y, z) in enumerate(pts3D):
        if z > 0:
            pts2D[i, 0] = fx * x / z + cx
            pts2D[i, 1] = fy * y / z + cy

    return pts2D
