import numpy as np
import cv2
from stereoVO.structures import StateBolts

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

class MatchingEngine():
    def __init__(self, prevInliers, prevDescriptors, currKpts, currDescriptors, params):
        self.prevInliers = prevInliers
        self.kp1 = prevInliers.left
        self.featuresA = prevDescriptors.left
        self.kp2 = currKpts.left
        self.featuresB = currDescriptors.left

        self.MIN_MATCH_COUNT = 10
        self.FLANN_INDEX_KDTREE = 1

    def process_tracked_features(self, ratio=0.5):
        # FLANN参数
        index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.featuresA, self.featuresB, k=2)

        good = []
        for m,n in matches:
            if m.distance < ratio*n.distance:
                good.append(m)
        self.good_matches = good

    def filter_inliers(self, pts3D_Filter):
        # import pdb; pdb.set_trace()
        if len(self.good_matches) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([ self.kp1[m.queryIdx] for m in self.good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ self.kp2[m.trainIdx].pt for m in self.good_matches ]).reshape(-1,1,2)

            # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, method=cv2.FM_RANSAC,ransacReprojThreshold=0.9, confidence=0.99)

            matchesMask = mask.ravel().tolist()
            # print(len(np.nonzero(matchesMask)[0]))
        else:
            print("Not enough matches are found - {}/{}".format(len(self.good_matches), self.MIN_MATCH_COUNT))
            matchesMask = None
            # Return empty results when not enough matches
            empty_pts = np.array([]).reshape(0, 2)
            return (empty_pts, empty_pts), (empty_pts, empty_pts), np.array([]).reshape(0, 3)
        
        selected_good = [m for m, cond in zip(self.good_matches, matchesMask) if cond>0]

        pointsTrackedLeft = []
        mask_epipolar = np.zeros(self.kp1.shape[0], dtype=bool)
        for m in selected_good:
            mask_epipolar[m.queryIdx] = True
            pointsTrackedLeft.append(self.kp2[m.trainIdx].pt)
        pointsTrackedLeft = np.array(pointsTrackedLeft)
        self.pointsTracked = StateBolts(pointsTrackedLeft, pointsTrackedLeft)        
        curr_pointsTracked = (self.pointsTracked.left, self.pointsTracked.right)
        prev_inliersTrackingFilter =  (self.prevInliers.left[mask_epipolar], self.prevInliers.left[mask_epipolar])
        pts3D_TrackingFilter = pts3D_Filter[mask_epipolar]
        prev_pts3D_TrackingFilter = pts3D_TrackingFilter
        
        # my_draw_matches(curr_pointsTracked[0], prev_inliersTrackingFilter[0], self.currFrames.left, self.prevFrames.left, "Tracking Matched")
        return prev_inliersTrackingFilter, curr_pointsTracked, prev_pts3D_TrackingFilter