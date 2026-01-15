"""
Depth Utility Functions for Stereo VO Pipeline

Functions to convert 2D keypoints to 3D points using pre-computed depth maps,
replacing the traditional stereo matching and triangulation steps.

Supports scaled depth maps where depth_map resolution differs from image resolution.
"""

import numpy as np

__all__ = ['get_3d_from_depth']


def get_3d_from_depth(keypoints_2d, depth_map, K, image_shape=None, interpolate=True):
    """
    Back-project 2D keypoints to 3D points using depth map.

    Supports depth maps with different resolution than the source image.
    When image_shape is provided, keypoint coordinates are scaled to depth map
    space and bilinear interpolation is used to get accurate depth values.

    Args:
        keypoints_2d: (N, 2) array of pixel coordinates (u, v) in image space
        depth_map: (H_d, W_d) depth map in meters (can be scaled, e.g., 0.35x image size)
        K: (3, 3) camera intrinsic matrix (for the original image resolution)
        image_shape: (H_img, W_img) tuple of original image dimensions.
                     If None, assumes depth_map has same size as image.
                     If provided, coordinates are scaled accordingly.
        interpolate: If True (default), use bilinear interpolation for depth lookup.
                     Recommended when using scaled depth maps.

    Returns:
        pts3D: (N, 3) array of 3D points in camera coordinate frame
        valid_mask: (N,) boolean array indicating valid points

    Example:
        # Image is 1920x1080, depth map is 672x378 (0.35 scale)
        pts3D, mask = get_3d_from_depth(
            keypoints_2d,      # coordinates in 1920x1080 space
            depth_map,         # shape (378, 672)
            K,                 # intrinsics for 1920x1080
            image_shape=(1080, 1920)  # (H, W) of original image
        )
    """
    if len(keypoints_2d) == 0:
        return np.array([]).reshape(0, 3), np.array([], dtype=bool)

    keypoints_2d = np.asarray(keypoints_2d)
    if keypoints_2d.ndim == 1:
        keypoints_2d = keypoints_2d.reshape(1, 2)

    N = keypoints_2d.shape[0]
    H_depth, W_depth = depth_map.shape[:2]

    # Calculate scale factors
    if image_shape is not None:
        H_img, W_img = image_shape
        scale_x = W_depth / W_img
        scale_y = H_depth / H_img
    else:
        # Assume same size
        scale_x = 1.0
        scale_y = 1.0
        H_img, W_img = H_depth, W_depth

    # Extract intrinsic parameters (for original image resolution)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pts3D = np.zeros((N, 3), dtype=np.float64)
    valid_mask = np.zeros(N, dtype=bool)

    for i, (u, v) in enumerate(keypoints_2d):
        # Scale keypoint coordinates to depth map space
        u_depth = u * scale_x
        v_depth = v * scale_y

        # Boundary check in depth map space
        if not (0 <= v_depth < H_depth and 0 <= u_depth < W_depth):
            continue

        # Get depth value using bilinear interpolation
        if interpolate:
            z = _bilinear_interpolate(depth_map, u_depth, v_depth)
        else:
            # Nearest neighbor
            u_int = int(round(u_depth))
            v_int = int(round(v_depth))
            u_int = min(max(u_int, 0), W_depth - 1)
            v_int = min(max(v_int, 0), H_depth - 1)
            z = depth_map[v_int, u_int]

        # Valid depth check
        if z > 0 and np.isfinite(z):
            # Back-project to 3D using original image coordinates and intrinsics
            # (u, v are in original image space, K is for original image)
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pts3D[i] = [x, y, z]
            valid_mask[i] = True

    return pts3D, valid_mask


def _bilinear_interpolate(depth_map, u, v):
    """
    Bilinear interpolation of depth value at sub-pixel location.

    Args:
        depth_map: (H, W) depth map
        u: x-coordinate (column) in depth map space
        v: y-coordinate (row) in depth map space

    Returns:
        Interpolated depth value
    """
    H, W = depth_map.shape[:2]

    # Clamp coordinates to valid range
    u = max(0, min(u, W - 1))
    v = max(0, min(v, H - 1))

    u0, v0 = int(u), int(v)
    u1, v1 = min(u0 + 1, W - 1), min(v0 + 1, H - 1)

    # Get the four corner depths
    d00 = depth_map[v0, u0]
    d01 = depth_map[v0, u1]
    d10 = depth_map[v1, u0]
    d11 = depth_map[v1, u1]

    # Bilinear interpolation weights
    wu = u - u0
    wv = v - v0

    # Check if all corners have valid depth for proper interpolation
    valid_corners = [d00 > 0, d01 > 0, d10 > 0, d11 > 0]

    if all(valid_corners):
        # All corners valid - do full bilinear interpolation
        d = (1 - wv) * ((1 - wu) * d00 + wu * d01) + wv * ((1 - wu) * d10 + wu * d11)
    elif any(valid_corners):
        # Some corners valid - use weighted average of valid corners only
        weights = []
        values = []

        if d00 > 0:
            weights.append((1 - wu) * (1 - wv))
            values.append(d00)
        if d01 > 0:
            weights.append(wu * (1 - wv))
            values.append(d01)
        if d10 > 0:
            weights.append((1 - wu) * wv)
            values.append(d10)
        if d11 > 0:
            weights.append(wu * wv)
            values.append(d11)

        total_weight = sum(weights)
        if total_weight > 0:
            d = sum(w * v for w, v in zip(weights, values)) / total_weight
        else:
            d = 0
    else:
        # No valid corners
        d = 0

    return d
