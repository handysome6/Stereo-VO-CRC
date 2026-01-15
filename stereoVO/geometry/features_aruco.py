import cv2
import numpy as np

REVERSE_ARUCO = True
DICT = cv2.aruco.DICT_4X4_250
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def match_2frame_aruco_corners(left_id, left_corners, right_id, right_corners):
    '''
    input:
        left_id: N, 
        left_corners: N, 4, 2
    return:
        left_corners_matched_sorted: M, 4, 2
        right_corners_matched_sorted: M, 4, 2
        left_id_matched_sorted: M,
    '''

    common_id = np.intersect1d(left_id, right_id)
    indices_left_id = np.isin(left_id, common_id)
    indices_right_id = np.isin(right_id, common_id)

    left_corners_matched = left_corners[indices_left_id]
    left_id_matched = left_id[indices_left_id]
    left_corners_matched_sorted = left_corners_matched[np.argsort(left_id_matched)]

    right_corners_matched = right_corners[indices_right_id]
    right_id_matched = right_id[indices_right_id]
    right_corners_matched_sorted = right_corners_matched[np.argsort(right_id_matched)]
    left_id_matched_sorted = np.sort(left_id_matched)
    return left_corners_matched_sorted, right_corners_matched_sorted, left_id_matched_sorted

def id2feats(ids):
    '''
    input:
        left_id: N, 1,
    return:
        features: N*4, 1
    '''
    ids = np.tile(ids.T, (4,1)) # [N, 4]
    ids[0] = ids[0]*10 + 1
    ids[1] = ids[1]*10 + 2
    ids[2] = ids[2]*10 + 3
    ids[3] = ids[3]*10 + 4
    return ids.T.reshape([ids.shape[0]*ids.shape[1], 1])

class ArucoDetector():
    def __init__(self, img, aruco_dict=DICT):
        """
        img: BGR image from opencv
        indexs_dict: containing aruco id and cooresponding corner number
            i.e. indexs_dict = {
                0: 1,
                1: 4
            }
        aruco_dict: cv provided dict definition
        """
        # self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = img
        if REVERSE_ARUCO:
            img_255 = np.full_like(self.img, 255)
            self.img = img_255 - self.img
        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)

        # direct the aruco
        self.corners, self.ids, rejects = cv2.aruco.detectMarkers(self.img, self.dictionary)
        ids = np.squeeze(self.ids)
        corners = np.squeeze(self.corners)

        self.detect_dict = {}
        for id, corners in zip(ids, corners):
            self.detect_dict[id] = corners
    
    def get_corners_ids(self):
        corners_lits = []
        for i in range(len(self.corners)):
            corner_refined = cv2.cornerSubPix(self.img, self.corners[i][0], (5, 5), (-1, -1), criteria)
            corners_lits.append(np.expand_dims(corner_refined, 0))
        return np.array(corners_lits)[:,0,:,:], self.ids[:,0]


class ArucoDetectionEngine():
    def __init__(self, left_frame, right_frame, params):
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.params = params

    def get_matching_keypoints(self):
        left_ad = ArucoDetector(self.left_frame)
        right_ad = ArucoDetector(self.right_frame)

        self.left_points, self.left_ids = left_ad.get_corners_ids()
        self.right_points, self.right_ids = right_ad.get_corners_ids()
        self.left_corners, self.right_corners, self.matched_ids = match_2frame_aruco_corners(self.left_ids, self.left_points, self.right_ids, self.right_points)

        matchedPoints = self.left_corners.reshape(-1, 2), self.right_corners.reshape(-1, 2)
        keypoints = self.left_points.reshape(-1, 2), self.right_points.reshape(-1, 2)

        descriptors_matched_Left = id2feats(np.expand_dims(np.array(self.matched_ids), -1))
        descriptors_matched_Right = descriptors_matched_Left
        matchedDescriptors = descriptors_matched_Left, descriptors_matched_Right

        descriptorsLeft = id2feats(np.expand_dims(np.array(self.left_ids), -1))
        descriptorsRight = id2feats(np.expand_dims(np.array(self.right_ids), -1))
        descriptors = descriptorsLeft, descriptorsRight

        return matchedPoints, keypoints, matchedDescriptors, descriptors


if __name__ == "__main__":
    left_path = "/home/liujunjie/HKCRC/update/rect_imgs/cam0/17168844339719872.jpg"
    right_path = "/home/liujunjie/HKCRC/update/rect_imgs/cam1/17168844339719872.jpg"
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    aruco_det = ArucoDetectionEngine(img_left, img_right, None)
    matchedPoints, keypoints, matchedDescriptors, descriptors = aruco_det.get_matching_keypoints()
    # print(matchedPoints[0].shape, matchedPoints[0].shape)
    # print(keypoints[0].shape, keypoints[0].shape)
    # print(matchedDescriptors[0], descriptors[0].shape)