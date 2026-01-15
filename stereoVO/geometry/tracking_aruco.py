import numpy as np
import cv2
from stereoVO.structures import StateBolts

def tracking_aruco(left_id, right_id):
    '''
    input:
        left_id: N, 
        right_id: N, 
    return:
        M, 2
    '''
    matching_indices = []
    for idx_A, element_A in enumerate(left_id):
        idx_B = np.where(right_id == element_A)[0]
        if len(idx_B) > 0:
            matching_indices.append([idx_A, idx_B[0]])
    return np.array(matching_indices)

class ArucoMatchingEngine():
    def __init__(self, prevInliers, prevDescriptors, currKpts, currDescriptors, params):
        self.prevInliers = prevInliers
        self.kp1 = prevInliers.left
        self.featuresA = prevDescriptors.left
        self.kp2 = currKpts.left
        self.featuresB = currDescriptors.left

    def tracking(self, pts3D_Filter):
        tracked_indices = tracking_aruco(self.featuresA, self.featuresB)
        pointsTrackedLeft = self.kp2[tracked_indices[:,1]]
        self.pointsTracked = StateBolts(pointsTrackedLeft, pointsTrackedLeft)
        curr_pointsTracked = (self.pointsTracked.left, self.pointsTracked.right)
        prev_inliersTrackingFilter =  (self.prevInliers.left[tracked_indices[:,0]], self.prevInliers.left[tracked_indices[:,0]])
        
        prev_pts3D_TrackingFilter = pts3D_Filter[tracked_indices[:, 0]]
        return prev_inliersTrackingFilter, curr_pointsTracked, prev_pts3D_TrackingFilter
