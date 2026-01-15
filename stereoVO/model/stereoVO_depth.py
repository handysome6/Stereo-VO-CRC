"""
Stereo Visual Odometry with Pre-computed Depth Maps

This module provides a modified VO pipeline that uses pre-computed depth maps
instead of traditional stereo matching and triangulation.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

from stereoVO.structures import VO_StateMachine
from stereoVO.optimization import get_minimization
from stereoVO.geometry import (DetectionEngine,
                               TrackingEngine,
                               MatchingEngine,
                               ArucoDetectionEngine,
                               ArucoMatchingEngine,
                               get_3d_from_depth)


def my_draw_matches(left_kpt, right_kpt, left_img, right_img, window_name="Matched"):
    """Visualization helper for feature matches."""
    kp1 = []
    kp2 = []
    goodMatch = []
    for i in range(left_kpt.shape[0]):
        kp1.append(cv2.KeyPoint(left_kpt[i, 0], left_kpt[i, 1], 2))
        kp2.append(cv2.KeyPoint(right_kpt[i, 0], right_kpt[i, 1], 2))
        goodMatch.append(cv2.DMatch(i, i, 1))
    img_show = cv2.drawMatches(left_img, kp1, right_img, kp2, goodMatch, None, flags=2)
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 1400, 600)
    cv2.imshow(window_name, img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class StereoVOWithDepth:
    """
    Stereo Visual Odometry using pre-computed depth maps.

    This class implements a simplified VO pipeline that:
    1. Detects features in the left image only
    2. Uses depth map to get 3D coordinates (skips stereo matching & triangulation)
    3. Tracks features between consecutive frames
    4. Estimates pose using P3P solver
    5. Optionally optimizes pose using non-linear least squares
    """

    def __init__(self, intrinsic, PL, PR, params):
        """
        Initialize the VO pipeline.

        Args:
            intrinsic: (3, 3) camera intrinsic matrix for left camera
            PL: (3, 4) left projection matrix
            PR: (3, 4) right projection matrix
            params: Configuration parameters (AttriDict)
        """
        self.intrinsic = intrinsic
        self.PL = PL
        self.PR = PR
        self.params = params

        # Set feature detection method
        if self.params.geometry.detection.method == "SIFT":
            self.feature_detection_method = 'sift'
            self.feature_matching_method = 'matching'
        elif self.params.geometry.detection.method == "ARUCO":
            self.feature_detection_method = 'aruco'
            self.feature_matching_method = 'aruco'
        else:
            raise NotImplementedError("Feature detector not implemented")

        # Depth filtering parameters (can be configured via params)
        self.depth_params = {
            'min_depth': getattr(self.params.geometry, 'depth_min', 0.5),
            'max_depth': getattr(self.params.geometry, 'depth_max', 30.0),
            'max_radius': getattr(self.params.geometry.triangulation, 'maxRadius', 50.0),
        }

        self.prevState = None
        self.currState = None

    def __call__(self, left_frame, right_frame, depth_map, state_num):
        """
        Process a stereo frame pair with depth map.

        Args:
            left_frame: Left camera image (grayscale)
            right_frame: Right camera image (grayscale, can be None)
            depth_map: (H, W) depth map in meters
            state_num: Frame number

        Returns:
            location: (3,) camera translation
            orientation: (3, 3) camera rotation matrix
            T_mat: (4, 4) transformation matrix
        """
        if state_num == 0:
            self._process_first_frame(left_frame, right_frame, depth_map, state_num)
            return self.prevState.location, self.prevState.orientation, self.prevState.T_mat
        else:
            self._process_continuous_frame(left_frame, right_frame, depth_map, state_num)

        print("Frame {} Processing Done ...................".format(state_num + 1))
        print("Current Location : X : {x}, Y = {y}, Z = {z}".format(
            x=self.currState.location[0],
            y=self.currState.location[1],
            z=self.currState.location[2]))

        self.prevState = self.currState

        return self.currState.location, self.currState.orientation, self.currState.T_mat

    def _process_first_frame(self, left_frame, right_frame, depth_map, state_num):
        """Process the first frame to initialize the VO state."""
        # Initialize the stereo state
        self.prevState = VO_StateMachine(state_num)
        self.prevState.frames = left_frame, right_frame
        self.prevState.depth_map = depth_map

        self.prevState.pointsTracked._left = None
        self.prevState.pointsTracked._right = None

        # Update state with feature detection and depth-based 3D points
        self._update_stereo_state_with_depth(self.prevState, depth_map)

        # Initialize pose at origin
        self.prevState.location = np.array(self.params.initial.location)
        self.prevState.orientation = np.array(self.params.initial.orientation)
        self.prevState.T_mat = np.eye(4)

    def _process_continuous_frame(self, left_frame, right_frame, depth_map, state_num):
        """Process subsequent frames."""
        # Initialize current state
        self.currState = VO_StateMachine(state_num)
        self.currState.frames = left_frame, right_frame
        self.currState.depth_map = depth_map

        # Update state with feature detection and depth-based 3D points
        self._update_stereo_state_with_depth(self.currState, depth_map)

        # Feature tracking from prevState to currState
        self._process_feature_tracking()

        # Check if we have enough tracked points for PnP
        if (self.prevState.pts3D_Tracking is None or
            len(self.prevState.pts3D_Tracking) < 4):
            print("Warning: Not enough tracked points for PnP, keeping previous pose")
            # Keep previous pose
            self.currState.orientation = self.prevState.orientation.copy()
            self.currState.location = self.prevState.location.copy()
            self.currState.relative_pose = np.eye(4)
            self.currState.T_mat = self.prevState.T_mat.copy()
            return

        # P3P solver to get relative pose
        r_mat, t_vec = self._solve_pnp()

        # Optional optimization
        if self.params.geometry.lsqsolver.enable:
            r_mat, t_vec = self._do_optimization(r_mat, t_vec)

        # Update pose: C_n = C_{n-1} * dT_{n-1}
        self.currState.orientation = self.prevState.orientation @ r_mat
        self.currState.location = self.prevState.orientation @ t_vec + self.prevState.location.reshape(-1, 1)
        self.currState.location = self.currState.location.flatten()

        # Store transformation matrix
        self.currState.relative_pose = np.eye(4)
        self.currState.relative_pose[:3, :3] = r_mat
        self.currState.relative_pose[:3, [3]] = t_vec
        self.currState.T_mat = self.prevState.T_mat @ np.linalg.inv(self.currState.relative_pose)

        print("Relative pose:\n", self.currState.relative_pose)

    def _update_stereo_state_with_depth(self, stereoState, depth_map):
        """
        Update stereo state using depth map instead of triangulation.

        This replaces the original _update_stereo_state method which used:
        - Stereo matching
        - Epipolar constraint filtering
        - DLT triangulation

        Now it simply:
        1. Detects features in left image
        2. Gets 3D points directly from depth map
        """
        print("Feature Detection (Depth Mode)")

        left_frame = stereoState.frames.left
        right_frame = stereoState.frames.right

        # Feature detection (only need left image features)
        if self.feature_detection_method == 'sift':
            detection_engine = DetectionEngine(left_frame, right_frame, self.params)
            stereoState.matchedPoints, stereoState.keyPoints, \
                stereoState.matchedDescriptors, stereoState.descriptors = \
                detection_engine.get_matching_keypoints()

            # Use left matched points for 3D reconstruction
            keypoints_2d = stereoState.matchedPoints.left
            descriptors = stereoState.matchedDescriptors.left

        elif self.feature_detection_method == 'aruco':
            detection_engine = ArucoDetectionEngine(left_frame, right_frame, self.params)
            stereoState.matchedPoints, stereoState.keyPoints, \
                stereoState.matchedDescriptors, stereoState.descriptors = \
                detection_engine.get_matching_keypoints()

            # Use left matched points for 3D reconstruction
            keypoints_2d = stereoState.matchedPoints.left
            descriptors = stereoState.matchedDescriptors.left

        # Get 3D points from depth map (replaces triangulation)
        # Pass image_shape to support scaled depth maps (e.g., depth at 0.35x resolution)
        image_shape = left_frame.shape[:2]  # (H, W)
        pts3D, valid_mask = get_3d_from_depth(keypoints_2d, depth_map, self.intrinsic,
                                               image_shape=image_shape, interpolate=True)

        # Filter based on depth constraints
        depth_valid = np.ones(len(pts3D), dtype=bool)
        for i, (x, y, z) in enumerate(pts3D):
            if not valid_mask[i]:
                depth_valid[i] = False
                continue

            # Depth range check
            if z < self.depth_params['min_depth'] or z > self.depth_params['max_depth']:
                depth_valid[i] = False
                continue

            # Radial distance check
            radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            if radius > self.depth_params['max_radius']:
                depth_valid[i] = False

        # Apply filter mask
        final_mask = valid_mask & depth_valid

        # Store filtered results
        stereoState.pts3D = pts3D
        stereoState.pts3D_Filter = pts3D[final_mask]
        stereoState.InliersFilter = (keypoints_2d[final_mask], keypoints_2d[final_mask])  # Left only mode
        stereoState.matchedDescriptors = (descriptors[final_mask], descriptors[final_mask])
        stereoState.ratioTriangulationFilter = np.sum(final_mask) / max(len(final_mask), 1)

        print(f"  Detected {len(keypoints_2d)} features, {np.sum(final_mask)} valid with depth")

        if self.params.debug.my_draw_matches and right_frame is not None:
            my_draw_matches(keypoints_2d[final_mask], keypoints_2d[final_mask],
                            left_frame, left_frame, "Features with Valid Depth")

    def _process_feature_tracking(self):
        """Track features between previous and current frames."""
        prevFrames = self.prevState.frames
        currFrames = self.currState.frames
        prevInliers = self.prevState.InliersFilter

        if self.feature_matching_method == 'tracking':
            tracker = TrackingEngine(prevFrames, currFrames, prevInliers,
                                     self.intrinsic, self.params)
            tracker.process_tracked_features()
            self.prevState.inliersTracking, self.currState.pointsTracked, \
                self.prevState.pts3D_Tracking = tracker.filter_inliers(self.prevState.pts3D_Filter)

        elif self.feature_matching_method == 'matching':
            tracker = MatchingEngine(self.prevState.InliersFilter,
                                     self.prevState.matchedDescriptors,
                                     self.currState.keyPoints,
                                     self.currState.descriptors, self.params)
            tracker.process_tracked_features()
            self.prevState.inliersTracking, self.currState.pointsTracked, \
                self.prevState.pts3D_Tracking = tracker.filter_inliers(self.prevState.pts3D_Filter)

        elif self.feature_matching_method == 'aruco':
            tracker = ArucoMatchingEngine(self.prevState.InliersFilter,
                                          self.prevState.matchedDescriptors,
                                          self.currState.keyPoints,
                                          self.currState.descriptors, self.params)
            self.prevState.inliersTracking, self.currState.pointsTracked, \
                self.prevState.pts3D_Tracking = tracker.tracking(self.prevState.pts3D_Filter)

        if self.params.debug.my_draw_matches:
            my_draw_matches(self.currState.pointsTracked.left,
                            self.prevState.inliersTracking.left,
                            currFrames.left, prevFrames.left, "Tracking Matched")

    def _solve_pnp(self):
        """Solve PnP to estimate relative camera pose."""
        args_pnpSolver = self.params.geometry.pnpSolver

        for i in range(args_pnpSolver.numTrials):
            _, r_vec, t_vec, idxPose = cv2.solvePnPRansac(
                self.prevState.pts3D_Tracking,
                self.currState.pointsTracked.left,
                self.intrinsic,
                np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            )

            r_mat, _ = cv2.Rodrigues(r_vec)

            try:
                idxPose = idxPose.flatten()
            except:
                import pdb
                pdb.set_trace()

            ratio = len(idxPose) / len(self.prevState.pts3D_Tracking)
            scale = np.linalg.norm(t_vec)

            if scale < args_pnpSolver.deltaT and ratio > args_pnpSolver.minRatio:
                break

        # Remove outliers
        self.currState.pointsTracked = (
            self.currState.pointsTracked.left[idxPose],
            self.currState.pointsTracked.right[idxPose]
        )
        self.prevState.P3P_pts3D = self.prevState.pts3D_Tracking[idxPose]

        return r_mat, t_vec

    def _do_optimization(self, r_mat, t_vec):
        """Non-linear least squares optimization of pose."""
        # Convert to camera coordinates
        t_vec = -r_mat.T @ t_vec
        r_mat = r_mat.T
        r_vec, _ = cv2.Rodrigues(r_mat)

        doF = np.concatenate((r_vec, t_vec)).flatten()

        optRes = least_squares(
            get_minimization, doF,
            method='lm', max_nfev=2000,
            args=(self.prevState.P3P_pts3D,
                  self.currState.pointsTracked.left,
                  self.currState.pointsTracked.right,
                  self.PL, self.PR)
        )

        opt_rvec_cam = (optRes.x[:3]).reshape(-1, 1)
        opt_tvec_cam = (optRes.x[3:]).reshape(-1, 1)
        opt_rmat_cam, _ = cv2.Rodrigues(opt_rvec_cam)

        r_mat = opt_rmat_cam.T
        t_vec = -opt_rmat_cam.T @ opt_tvec_cam

        return r_mat, t_vec
