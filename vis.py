import numpy as np
import open3d as o3d

def draw_cam_poses(cam_poses): 
    '''
        cam_poses: list
    '''
    WIDTH = 1280
    HEIGHT = 720

    visual_cam_intrinsics = np.array([
        [1.16962109e+03, 0.00000000e+00, 6.46295044e+02, 0.00000000e+00],
        [0.00000000e+00, 1.16710510e+03, 4.89927032e+02, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])

    show_cams = []
    for cam_pose in cam_poses:
        cam_pose_show = np.linalg.inv(cam_pose) # 如果不取逆,显示出来的位姿和实际的是相反的, 因为open3d显示的pose是源坐标系的点投影到目标坐标系的变换
        camera_Lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH,
                                                                    view_height_px=HEIGHT,
                                                                    intrinsic=visual_cam_intrinsics[:3, :3], extrinsic=cam_pose_show, scale=0.5)
        camera_Lines.paint_uniform_color([1, 0.1, 0])

        show_cams.append(camera_Lines)
    return show_cams

def show_cam_poses(cam_poses): 
    WIDTH = 1280
    HEIGHT = 720

    visual_cam_intrinsics = np.array([
        [1.16962109e+03, 0.00000000e+00, 6.46295044e+02, 0.00000000e+00],
        [0.00000000e+00, 1.16710510e+03, 4.89927032e+02, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vizualizer = o3d.visualization.Visualizer()
    vizualizer.create_window(width=1200, height=800)
    vizualizer.add_geometry(axis_pcd)

    for cam_pose in cam_poses:
        cam_pose_show = np.linalg.inv(cam_pose) # 如果不取逆,显示出来的位姿和实际的是相反的, 因为open3d显示的pose是源坐标系的点投影到目标坐标系的变换
        camera_Lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH,
                                                                    view_height_px=HEIGHT,
                                                                    intrinsic=visual_cam_intrinsics[:3, :3], extrinsic=cam_pose_show, scale=0.5)
        camera_Lines.paint_uniform_color([1, 0.1, 0])
        vizualizer.add_geometry(camera_Lines)

    vizualizer.run()