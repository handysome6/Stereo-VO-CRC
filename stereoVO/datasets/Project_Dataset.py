"""
Project Dataset Loader for Stereo VO Pipeline

Loads stereo image pairs from the project structure where each timestamp folder contains:
    {timestamp}/
        rect_left.jpg      - Rectified left image
        rect_right.jpg     - Rectified right image
        depth_meter.npy    - Depth map in meters
        cloud.ply          - Point cloud
        K.txt              - Camera intrinsics
        camera_model.json  - Camera calibration
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path

__all__ = ['ProjectDataset', 'ProjectFileNames']


def _imread_unicode(filepath):
    """
    Read image from path that may contain Unicode characters (e.g., Chinese).

    cv2.imread() on Windows cannot handle non-ASCII paths directly.
    This workaround reads file bytes with numpy and decodes with cv2.imdecode().

    Args:
        filepath: Path to image file (str or Path)

    Returns:
        Image as numpy array, or None if file cannot be read
    """
    try:
        # Read file as bytes using numpy (handles Unicode paths)
        with open(str(filepath), 'rb') as f:
            img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        # Decode image from bytes
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Warning: Failed to read image {filepath}: {e}")
        return None


class ProjectFileNames:
    """Standardized file names for the project structure."""
    RAW_LEFT = "raw_left.jpg"
    RAW_RIGHT = "raw_right.jpg"
    RECT_LEFT = "rect_left.jpg"
    RECT_RIGHT = "rect_right.jpg"
    CAMERA_MODEL = "camera_model.json"
    CLOUD_PLY = "cloud.ply"
    DEPTH_NPY = "depth_meter.npy"
    DISP_NPY = "disp.npy"
    K_TXT = "K.txt"
    IMG0 = "img0.jpg"
    IMG1 = "img1.jpg"
    VIS_PNG = "vis.png"
    Q_NPY = "Q.npy"


class ProjectDataset:
    """
    Dataset loader for project structure with pre-computed depth maps.

    Each frame folder contains rectified stereo images and corresponding
    depth information, eliminating the need for stereo matching.
    """

    def __init__(self, project_path, intrinsic_l=None, intrinsic_r=None, extrinsic=None):
        """
        Initialize the dataset loader.

        Args:
            project_path: Path to the project folder containing timestamp subfolders
            intrinsic_l: Left camera intrinsic matrix (3x3), if None will load from K.txt
            intrinsic_r: Right camera intrinsic matrix (3x3), if None uses intrinsic_l
            extrinsic: Extrinsic matrix (4x4), right camera relative to left
        """
        self.project_path = Path(project_path)

        # Get all timestamp folders sorted
        self.frame_folders = self._load_frame_folders()

        if len(self.frame_folders) == 0:
            raise ValueError(f"No valid frame folders found in {project_path}")

        # Load camera parameters from first frame if not provided
        first_frame_path = self.frame_folders[0]

        if intrinsic_l is None:
            self.intrinsic_l = self._load_intrinsic(first_frame_path)
        else:
            self.intrinsic_l = np.array(intrinsic_l)

        if intrinsic_r is None:
            self.intrinsic_r = self.intrinsic_l.copy()
        else:
            self.intrinsic_r = np.array(intrinsic_r)

        if extrinsic is None:
            self.extrinsic = self._load_extrinsic(first_frame_path)
        else:
            self.extrinsic = np.array(extrinsic)

        # Compute projection matrices
        left_pose = np.eye(4)[:3, :]
        self.PL = self.intrinsic_l @ left_pose
        self.PR = self.intrinsic_r @ self.extrinsic[:3, :]

        print(f"ProjectDataset initialized with {len(self.frame_folders)} frames")

    def _load_frame_folders(self):
        """Load and sort all valid frame folders."""
        folders = []

        for item in self.project_path.iterdir():
            if item.is_dir():
                # Check if folder contains required files
                rect_left = item / ProjectFileNames.RECT_LEFT
                depth_npy = item / ProjectFileNames.DEPTH_NPY

                if rect_left.exists() and depth_npy.exists():
                    folders.append(item)

        # Sort by folder name (timestamp)
        folders.sort(key=lambda x: x.name)
        return folders

    def _load_intrinsic(self, frame_path):
        """Load camera intrinsic matrix from K.txt or camera_model.json."""
        k_file = frame_path / ProjectFileNames.K_TXT
        camera_model_file = frame_path / ProjectFileNames.CAMERA_MODEL

        if k_file.exists():
            # K.txt format: first line is 9 numbers (3x3 matrix), second line is baseline
            with open(k_file, 'r') as f:
                lines = f.readlines()

            if len(lines) >= 1:
                # Parse first line as intrinsic matrix (9 numbers)
                k_values = [float(x) for x in lines[0].strip().split()]
                if len(k_values) == 9:
                    K = np.array(k_values).reshape(3, 3)
                    print(f"Loaded intrinsic from K.txt:\n{K}")
                    return K

            # Fallback: try loading as simple matrix
            K = np.loadtxt(str(k_file))
            if K.shape == (3, 3):
                return K
            elif K.size == 9:
                return K.reshape(3, 3)

        if camera_model_file.exists():
            with open(camera_model_file, 'r') as f:
                camera_data = json.load(f)

            # Try to extract intrinsic from camera_model.json
            if 'K' in camera_data:
                return np.array(camera_data['K']).reshape(3, 3)
            elif 'intrinsic' in camera_data:
                return np.array(camera_data['intrinsic']).reshape(3, 3)
            elif 'cm1' in camera_data:
                # Use rectified intrinsic (cm1 for left camera)
                return np.array(camera_data['cm1'])
            elif 'fx' in camera_data:
                fx = camera_data['fx']
                fy = camera_data.get('fy', fx)
                cx = camera_data['cx']
                cy = camera_data['cy']
                return np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])

        raise ValueError(f"Cannot load intrinsic matrix from {frame_path}")

    def _load_extrinsic(self, frame_path):
        """Load extrinsic matrix from K.txt or camera_model.json."""
        k_file = frame_path / ProjectFileNames.K_TXT
        camera_model_file = frame_path / ProjectFileNames.CAMERA_MODEL

        # First try to load baseline from K.txt (second line)
        baseline = None
        if k_file.exists():
            with open(k_file, 'r') as f:
                lines = f.readlines()
            if len(lines) >= 2:
                try:
                    baseline = float(lines[1].strip())
                    print(f"Loaded baseline from K.txt: {baseline} meters")
                except ValueError:
                    pass

        # If baseline found, create extrinsic with identity rotation
        if baseline is not None:
            extrinsic = np.eye(4)
            extrinsic[0, 3] = -baseline  # Right camera is baseline to the right
            print(f"Extrinsic matrix (from baseline):\n{extrinsic}")
            return extrinsic

        # Try camera_model.json
        if camera_model_file.exists():
            with open(camera_model_file, 'r') as f:
                camera_data = json.load(f)

            if 'extrinsic' in camera_data:
                return np.array(camera_data['extrinsic']).reshape(4, 4)

            # Build extrinsic from R and T
            if 'R' in camera_data and 'T' in camera_data:
                R = np.array(camera_data['R'])
                T = np.array(camera_data['T'])

                # T is typically in millimeters, convert to meters
                if np.abs(T[0]) > 10:  # If T[0] > 10, likely in mm
                    T = T / 1000.0
                    print(f"Converted T from mm to meters: {T}")

                extrinsic = np.eye(4)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = T
                print(f"Extrinsic matrix (from R, T):\n{extrinsic}")
                return extrinsic

            elif 'baseline' in camera_data:
                baseline = camera_data['baseline']
                extrinsic = np.eye(4)
                extrinsic[0, 3] = -baseline
                return extrinsic

        # Default: identity with no translation (will need to be configured)
        print("Warning: No extrinsic found, using identity matrix")
        return np.eye(4)

    def __len__(self):
        """Return number of frames in dataset."""
        return len(self.frame_folders)

    def __getitem__(self, index):
        """
        Get frame data by index.

        Args:
            index: Frame index

        Returns:
            img_left: Rectified left image (grayscale)
            img_right: Rectified right image (grayscale)
            frame_path: Path to the frame folder (string)
        """
        frame_path = self.frame_folders[index]

        # Load rectified images (use _imread_unicode for paths with non-ASCII characters)
        left_path = frame_path / ProjectFileNames.RECT_LEFT
        right_path = frame_path / ProjectFileNames.RECT_RIGHT

        img_left = _imread_unicode(left_path)
        img_right = _imread_unicode(right_path) if right_path.exists() else None

        # Convert to grayscale
        if img_left is not None and len(img_left.shape) == 3:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        if img_right is not None and len(img_right.shape) == 3:
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        print(f"=================== Frame {index} =============== {frame_path.name}")

        return img_left, img_right, str(frame_path)

    def get_depth(self, index):
        """
        Get depth map for a frame.

        Args:
            index: Frame index

        Returns:
            depth_map: (H, W) depth map in meters
        """
        frame_path = self.frame_folders[index]
        depth_path = frame_path / ProjectFileNames.DEPTH_NPY

        if not depth_path.exists():
            raise FileNotFoundError(f"Depth map not found: {depth_path}")

        return np.load(str(depth_path))

    def get_disparity(self, index):
        """
        Get disparity map for a frame.

        Args:
            index: Frame index

        Returns:
            disparity_map: (H, W) disparity map
        """
        frame_path = self.frame_folders[index]
        disp_path = frame_path / ProjectFileNames.DISP_NPY

        if not disp_path.exists():
            return None

        return np.load(str(disp_path))

    def get_point_cloud(self, index):
        """
        Get point cloud for a frame (if needed for verification).

        Args:
            index: Frame index

        Returns:
            points: (N, 3) point cloud array, or None if not available
        """
        frame_path = self.frame_folders[index]
        cloud_path = frame_path / ProjectFileNames.CLOUD_PLY

        if not cloud_path.exists():
            return None

        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(cloud_path))
            return np.asarray(pcd.points)
        except ImportError:
            print("Warning: open3d not available, cannot load point cloud")
            return None

    def get_frame_info(self, index):
        """
        Get all available data for a frame.

        Args:
            index: Frame index

        Returns:
            dict with keys: img_left, img_right, depth, disparity, frame_path
        """
        img_left, img_right, frame_path = self[index]

        return {
            'img_left': img_left,
            'img_right': img_right,
            'depth': self.get_depth(index),
            'disparity': self.get_disparity(index),
            'frame_path': frame_path
        }
