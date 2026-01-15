import cv2
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['DetectionEngine']

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

class DetectionEngine():

    """
    Main Engine Code for detection of feature in the frames and 
    matching of features in the frames at the current stereo state
    """
    
    def __init__(self, left_frame, right_frame, params):

        """
        :param left_frame (np.array): of size (HxWx3 or HxW) of stereo configuration
        :param right_frame (np.array): of size (HxWx3 or HxW) of stereo configuation
        :param params (AttriDict): contains parameters for the stereo configuration, 
                                   detection of and matching of features                              computer vision features
        """
        
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.params = params
        
    def get_matching_keypoints(self):

        """
        Runs a feature detector on both the features, computes keypoints and descriptor
        information and performs flann based matching to match keypoints across the left
        and right frames and applying ratio test to filter "good" matching features

        Returns:
            matchedPoints (tuple of np.array for (left, right)): each of size (N,2) of matched features
            keypoints (tuple of lists for (left, right)) : each list containing metadata of kepoints from feature detector
            descriptors (tuple of np.array for (left, right)) : each of size (MXd) list containing feature vector of keypoints from feature detector 
        """
    
        if self.params.geometry.detection.method == "SIFT":
            detector = cv2.SIFT_create()
        else:
            raise NotImplementedError("Feature Detector has not been implemented. Please refer to the Contributing guide and raise a PR")

        if len(self.left_frame.shape) == 3:
            self.left_frame = cv2.cvtColor(self.left_frame.left)

        if len(self.right_frame.shape) == 3:
            self.right_frame = cv2.cvtColor(self.right_frame)

        keyPointsLeft, descriptorsLeft = detector.detectAndCompute(self.left_frame, None)
        keyPointsRight, descriptorsRight = detector.detectAndCompute(self.right_frame, None)

        if self.params.debug.plotting.features:
            DetectionEngine.plot_feature(self.left_frame, self.right_frame, keyPointsLeft, keyPointsRight)

        args_feature_matcher = self.params.geometry.featureMatcher.configs

        indexParams = args_feature_matcher.indexParams
        searchParams = args_feature_matcher.searchParams

        if self.params.geometry.featureMatcher.method == "FlannMatcher":
            matcher = cv2.FlannBasedMatcher(indexParams, searchParams)
        else:
            raise NotImplementedError("Feature Matcher has not been implemented. Please refer to the Contributing guide and raise a PR")

        matches = matcher.knnMatch(descriptorsLeft, descriptorsRight, args_feature_matcher.K)

        #Apply ratio test 
        goodMatches = []
        ptsLeft = []
        ptsRight = []
        descriptors_matched_Left = []
        descriptors_matched_Right = []

        for m, n in matches:  
            if m.distance < args_feature_matcher.maxRatio * n.distance: 
                goodMatches.append([m]) 
                ptsLeft.append(keyPointsLeft[m.queryIdx].pt) 
                ptsRight.append(keyPointsRight[m.trainIdx].pt)

        ptsLeft = np.array(ptsLeft).astype('float64')
        ptsRight = np.array(ptsRight).astype('float64')

        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, method=cv2.FM_RANSAC,ransacReprojThreshold=0.9, confidence=0.99)
        matchesMask = mask.ravel().tolist()
        filtered_goodMatches = [m[0] for m, cond in zip(goodMatches, matchesMask) if cond>0]

        filtered_ptsLeft = []
        filtered_ptsRight = []
        for m in filtered_goodMatches:
            filtered_ptsLeft.append(keyPointsLeft[m.queryIdx].pt)
            filtered_ptsRight.append(keyPointsRight[m.trainIdx].pt)
            descriptors_matched_Left.append(descriptorsLeft[m.queryIdx])
            descriptors_matched_Right.append(descriptorsRight[m.trainIdx])

        ptsLeft = np.array(filtered_ptsLeft).astype('float64')
        ptsRight = np.array(filtered_ptsRight).astype('float64')
        descriptors_matched_Left = np.array(descriptors_matched_Left)
        descriptors_matched_Right = np.array(descriptors_matched_Right)
        goodMatches = filtered_goodMatches

        if self.params.debug.plotting.featureMatches:
            DetectionEngine.plot_feature_matches(self.left_frame,  self.right_frame, keyPointsLeft, keyPointsRight, goodMatches)

        matchedPoints = ptsLeft, ptsRight
        matchedDescriptors = descriptors_matched_Left, descriptors_matched_Right
        keypoints = keyPointsLeft, keyPointsRight
        descriptors = descriptorsLeft, descriptorsRight
        # my_draw_matches(ptsLeft, ptsRight, self.left_frame, self.right_frame, "Stereo Matched")

        return matchedPoints, keypoints, matchedDescriptors, descriptors
    
    @staticmethod
    def plot_feature(left_frame, right_frame, keyPointsLeft, keyPointsRight):
        
        """
        Helper function for plotting features on respective left and right frames
        usign keypoint computed from feature detector
        """
        
        kp_on_left_frame = cv2.drawKeypoints(left_frame,
                                             keyPointsLeft,
                                             None)

        kp_on_right_frame = cv2.drawKeypoints(right_frame, 
                                              keyPointsRight,
                                              None)

        plt.figure(figsize=(30, 15))

        plt.subplot(1, 2, 1)
        plt.imshow(kp_on_left_frame)

        plt.subplot(1, 2, 2)
        plt.imshow(kp_on_right_frame)

        plt.show()

    @staticmethod
    def plot_feature_matches(left_frame, right_frame, keyPointsLeft, keyPointsRight, matches):

        """
        Helper function for plotting feature matches across the left and right frames
        using keypoint calculated from the feature detector and matches from the featue matcher
        """

        feature_matches = cv2.drawMatchesKnn(left_frame,
                                             keyPointsLeft,
                                             right_frame,
                                             keyPointsRight,
                                             matches,
                                             outImg=None,
                                             flags=0)
        plt.figure(figsize=(20, 10))
        plt.imshow(feature_matches)
        plt.show()
