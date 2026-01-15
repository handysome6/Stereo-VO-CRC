import numpy as np
import cv2
from stereoVO.structures import StateBolts
from stereoVO.geometry import filter_matching_inliers

def my_draw_matches(left_kpt, right_kpt, left_img, right_img, window_name="Matched"):
    kp1 = []
    kp2 = []
    goodMatch = []
    for i in range(left_kpt.shape[0]):
        kp1.append(cv2.KeyPoint(left_kpt[i,0], left_kpt[i,1], 2))
        kp2.append(cv2.KeyPoint(right_kpt[i,0], right_kpt[i,1], 2))
        goodMatch.append(cv2.DMatch(i, i, 1))
    img_show = cv2.drawMatches(left_img, kp1, right_img, kp2, goodMatch, None, flags=2)
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 1400, 600)  # 设置窗口大小
    cv2.imshow(window_name, img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
To Do: Make the tracking engine modular to accept the following 
       set of parameters for optical flow as inputs
'''
lk_params = dict(winSize  = (21, 21), 
                 maxLevel = 5,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))


class TrackingEngine():

    """
    Main Engine code for tracking features across two time instances
    (called prevState and currState here)
    """

    def __init__(self, prevFrames, currFrames, prevInliers, intrinsic, params):

        """
        :param prevFrames  (StateBolts::(np.array, np.array):  
                           (size(H,W), size(H,w)) frames of previous stero state wrapped in StateBolts Module
        :param currFrames  (StateBolts::(np.array, np.array):  
                           (size(H,W), size(H,w)) frames of current stero state wrapped in StateBolts Module
        :param prevInliers (StateBolts::(np.array, np.array): 
                           (size(N,2), size(N,2)) detected and matched  keypoints (filtered) on left and right frame
        :param intrinsic   (np.array) : size(3,3) : camera calibration matrix
        :param params      (AttriDict): contains parameters for the stereo configuration, 
                                        detection of features, tracking and other geomtric
                                        computer vision features
        """
        self.prevFrames = prevFrames
        self.currFrames = currFrames
        self.prevInliers = prevInliers
        self.intrinsic = intrinsic
        self.params = params

    def process_tracked_features(self):

        """
        Executor module which tracks the features on the current set of frames
        and filters inliers from the detection engine and tracking engine
        """

        # Track features on left and right frame in the current state from previous states 
        _, pointsTrackedLeft, maskTrackingLeft = TrackingEngine.track_features(self.prevFrames.left,
                                                                               self.currFrames.left,
                                                                               self.prevInliers.left)

        # Only Left
        self.maskTracking = maskTrackingLeft
        self.pointsTracked = StateBolts(pointsTrackedLeft[self.maskTracking], pointsTrackedLeft[self.maskTracking])
        self.prevInliers = StateBolts(self.prevInliers.left[self.maskTracking], self.prevInliers.left[self.maskTracking])
        my_draw_matches(self.pointsTracked.left, self.prevInliers.left, self.currFrames.left, self.prevFrames.left, "Tracking before filtering")

    def filter_inliers(self, pts3D_Filter):

        """
        Executor module to apply epipolar contraint on the prevstate and current state frames
        and furtker filter inliers from detection engine, trackign engine and 3D points
        """

        # Remove non-valid points from inliers filtered in prev state
        pts3D_TrackingFilter = pts3D_Filter[self.maskTracking]
        
        # Remove Outliers using Epipolar Geometry (RANSAC)
        (_, __), mask_epipolar_left = filter_matching_inliers(self.prevInliers.left, self.pointsTracked.left, self.intrinsic, self.params)
        
        # Only Left
        mask_epipolar = mask_epipolar_left
        
        # Remove Outliers from tracked points correspondences on both stereo states and 3D points
        print("self.pointsTracked.left before: ", self.pointsTracked.left.shape)
        curr_pointsTracked = (self.pointsTracked.left[mask_epipolar], self.pointsTracked.right[mask_epipolar])
        print("self.pointsTracked.left after: ", curr_pointsTracked[0].shape)
        prev_inliersTrackingFilter =  (self.prevInliers.left[mask_epipolar], self.prevInliers.right[mask_epipolar])
        prev_pts3D_TrackingFilter = pts3D_TrackingFilter[mask_epipolar] 

        my_draw_matches(curr_pointsTracked[0], prev_inliersTrackingFilter[0], self.currFrames.left, self.prevFrames.left, "Tracking Matched")
        
        return prev_inliersTrackingFilter, curr_pointsTracked, prev_pts3D_TrackingFilter

    @staticmethod
    def track_features(imageRef, imageCur, pointsRef):

        """
        :param imageRef  (np.array): size(H,W) grayscale image as reference image to track feature
        :param imageCur  (np.array): size(H,W) grayscale image as current image to track features on
        :param pointsRef (np.array): size(N,2) keypoints to be used as reference for tracking 

        Returns
            pointsRef    (np.array): size(N,2) same as param pointsRef
            points_t0_t1 (np.array): size(N,2) tracked features/keypoints on current frame 
            mask_t0_t1   (np.array): size(N) indexes for "good" tracking points
        """

        # Do asserion test on the input
        assert len(pointsRef.shape) == 2 and pointsRef.shape[1] == 2

        # Reshape input and track features
        pointsRef = pointsRef.reshape(-1, 1, 2).astype('float32')
        points_t0_t1, mask_t0_t1, _ = cv2.calcOpticalFlowPyrLK(imageRef, 
                                                               imageCur,
                                                               pointsRef, 
                                                               None, 
                                                               **lk_params)
        # import pdb; pdb.set_trace()
        # my_draw_matches(pointsRef[:,0,:], points_t0_t1[:,0,:], imageRef, imageCur, "Tracking")

        # Reshape ouput and return output and mask for tracking points
        pointsRef = pointsRef.reshape(-1,2)
        points_t0_t1 = points_t0_t1.reshape(-1,2)
        mask_t0_t1 = mask_t0_t1.flatten().astype(bool)

        return pointsRef, points_t0_t1, mask_t0_t1
