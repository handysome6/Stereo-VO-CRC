import cv2
import os
import numpy as np
from pathlib import Path

__all__ = ['CRCDataset']


class CameraParameters():
    def __init__(self, fx, fy, cx, cy):

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @property
    def camera_matrix(self):

        matrix = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fx, self.cy],
                           [0.0, 0.0, 1.0]])
        return matrix

    def __call__(self):

        return self.camera_matrix


class CRCDataset():
    def __init__(self, path, intrinsic_l, intrinsic_r, extrinsic):

        """
        param: path (str):path to images
        """

        self.path = Path(path)

        self.left_images_path = str(self.path.joinpath('cam0'))
        self.right_images_path = str(self.path.joinpath('cam1'))

        self.left_image_paths = self.load_image_paths(self.left_images_path)
        self.right_image_paths = self.load_image_paths(self.right_images_path)

        # rectified 5.28
        # self.intrinsic = np.array([[5.03535962e+03, 0.00000000e+00, 2.84703403e+03],
        #                             [0.00000000e+00, 5.03813141e+03, 1.88368015e+03],
        #                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.extrinsic = np.array([[ 1.00000000e+00, 2.61790114e-14, 1.29561976e-14, -9.25217755e-01],
        #                             [-2.61784769e-14, 1.00000000e+00, -2.57468088e-15, 1.13502415e-19],
        #                             [-1.29560966e-14, 2.57479693e-15, 1.00000000e+00, 4.33680869e-19],
        #                             [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # undist 5.28
        # self.intrinsic = np.array([[5.03538941e+03, 0.00000000e+00, 2.75641007e+03],
        #                             [0.00000000e+00, 5.03814796e+03, 1.78567047e+03],
        #                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.intrinsic_r = np.array([[5.04268658e+03, 0.00000000e+00, 2.71600192e+03],
        #                             [0.00000000e+00, 5.04354287e+03, 1.83305351e+03],
        #                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.extrinsic = np.array([[ 0.99994233, 0.00893681, 0.00595496, -0.92521241],
        #                             [-0.00892883, 0.9999592, -0.00136546, 0.00200932],
        #                             [-0.00596692, 0.00131221, 0.99998134, 0.00241891],
        #                             [0.,          0.,          0.,          1.,        ]])


        # # for undist images 6.12 data
        # self.intrinsic = np.array([[5.03327197e+03, 0.00000000e+00, 2.77428642e+03],
        #                             [0.00000000e+00, 5.03533169e+03, 1.78327701e+03],
        #                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.intrinsic_r = np.array([[5.04701629e+03, 0.00000000e+00, 2.71958253e+03],
        #                             [0.00000000e+00, 5.04762944e+03, 1.80143530e+03],
        #                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.extrinsic = np.array([[ 0.9999535, 0.00641968, 0.00719584, -0.92583359],
        #                             [-0.00642832, 0.99997864, 0.00117817, 0.00139852],
        #                             [-0.00718812, -0.00122437, 0.99997342, 0.00566296],
        #                             [ 0.,         0.,         0.,         1.,       ]])

        # # for rectified images 6.12 data
        # self.intrinsic = np.array([[5.03321669e+03, 0.00000000e+00, 2.86035174e+03],
        #                         [0.00000000e+00, 5.03534363e+03, 1.88263784e+03],
        #                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.intrinsic_r = np.array([[5.03321669e+03, 0.00000000e+00, 2.86035174e+03],
        #                         [0.00000000e+00, 5.03534363e+03, 1.88263784e+03],
        #                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.extrinsic = np.array([[1.00000000e+00, 5.71419636e-20, -1.97538368e-19, -9.25851966e-01],
        #                         [-8.09845113e-19, 1.00000000e+00, 2.25347847e-19, 3.69306365e-19],
        #                         [ 5.50186412e-19, -5.74965332e-20, 1.00000000e+00, -8.67361738e-19],
        #                         [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # for rectified images 5.28 data
        self.intrinsic_l = intrinsic_l
        self.intrinsic_r = intrinsic_r
        self.extrinsic = extrinsic
        
        left_pose = np.eye(4)[:3,:]
        self.PL = self.intrinsic_l @ left_pose
        self.PR = self.intrinsic_r @ self.extrinsic[:3,:]


    def load_image_paths(self, image_dir):

        """
        Returns images in path sorted by the frame number

        :path image_dir(str)     : path to image dir
        :return img_paths (list) : image paths sorted by frame number
        """
        img_paths = [os.path.join(image_dir, img_id) for img_id in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, img_id))]
        img_paths.sort()

        return img_paths

    def __len__(self):

        """
        Fetch number of images returned by the dataloader
        """
        
        return len(self.left_image_paths)

    def __getitem__(self, index):

        """
        Fetches left frame, right frame and ground pose for a particular frame number (or time instant)

        :param index(int): frame number (index of stereo state)

        Returns
            :img_left  (np.array): size(H,W) left frame of a stereo configuration for a particular frame number  
            :img_right (np.array): size(H,W) right frame of a stereo configuration for a particular frame number
        """
        print("===================left_frame===============", self.left_image_paths[index])

        img_left = cv2.imread(self.left_image_paths[index])
        img_right = cv2.imread(self.right_image_paths[index])

        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        return img_left, img_right, self.left_image_paths[index]
