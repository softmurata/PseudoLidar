import open3d as o3d
import os
import numpy as np
import subprocess

# camera configuration
class Config(object):

    def __init__(self):
        self.width = 640
        self.height = 480
        self.fx = 381.694
        self.fy = 381.694
        self.cx = 323.56
        self.cy = 237.11
        self.bx = 0
        self.by = 0

# 3D match utils

def create_point_cloud_from_rgbd(rgb_path, depth_path, voxel_size, config):
    # create target point cloud
    rgb = o3d.io.read_image(rgb_path)
    depth = o3d.io.read_image(depth_path)
    
    # set camera intrinsic parameters
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(config.width, config.height, config.fx, config.fy, config.cx, config.cy)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    return target_point_cloud

def prepare_dataset(point_cloud, voxel_size):
    # calculate keypoints(by voxel down sampling)
    keypoints = point_cloud.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    keypoints.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # calculate fpfh(?)
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.registration.compute_fpfh_feature(keypoints, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
    
    return keypoints, pcd_fpfh

def execute_global_registration(
        source_down, target_down, reference_desc, target_desc, distance_threshold):

    result = o3d.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, reference_desc, target_desc,
            distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(source, target, distance_threshold,
            result_ransac.transformation,
            o3d.registration.TransformationEstimationPointToPlane())
    return result

def save_down_sample_and_features(point_cloud_files, args):
    voxel_size = args.voxel_size
    output_dim = args.output_dim  # NN
    source_pcd = o3d.io.read_point_cloud(point_cloud_files[0])
    target_pcd = o3d.io.read_point_cloud(point_cloud_files[1])
    source_down, source_fpfh = prepare_dataset(source_pcd, voxel_size)
    target_down, target_fpfh = prepare_dataset(target_pcd, voxel_size)
    
    print()
    print('--result part--')
    source_points = np.asarray(source_down.points)
    source_colors = np.asarray(source_down.colors)
    source_features = np.asarray(source_fpfh.data)
    
    target_points = np.asarray(target_down.points)
    target_colors = np.asarray(target_down.colors)
    target_features = np.asarray(target_fpfh.data)
    
    ds_save_dir = './SmnetData/down_sample/'
    desc_save_dir = './data/demo/32_dim/'
    
    os.makedirs(ds_save_dir, exist_ok=True)    
    os.makedirs(desc_save_dir, exist_ok=True)
    
    source_down_sample_pcd_path = ds_save_dir + 'source.ply'
    target_down_sample_pcd_path = ds_save_dir + 'target.ply'
    
    # down sampling point cloud
    o3d.io.write_point_cloud(source_down_sample_pcd_path, source_down)
    o3d.io.write_point_cloud(target_down_sample_pcd_path, target_down)
    
    down_point_cloud_files = [source_down_sample_pcd_path, target_down_sample_pcd_path]
    
    # fpfh
    source_desc_path = desc_save_dir + 'source.npz'
    target_desc_path = desc_save_dir + 'target.npz'
    np.savez(source_desc_path, source_features)
    np.savez(target_desc_path, target_features)
    
    desc_files = [source_desc_path, target_desc_path]
    print()
    print('----confirmation-----')
    print(source_points.shape)
    print(source_colors.shape)
    print(source_features.shape)
    
    return down_point_cloud_files, desc_files

def create_down_pcd(point_cloud_files, voxel_size, output_folder):
    
    for i in range(len(point_cloud_files)):
        args = "./3DSmoothNet -f" + point_cloud_files[i] + " -r" + str(voxel_size) + " -o {}".format(output_folder)
        subprocess.call(args, shell=True)
        
def inference(output_dim, input_folder, output_folder):
    """
    python main_cnn.py --run_mode=test --output_dim=32 --evaluate_input_folder='./data/demo/test' --evaluate_output_folder='./data/demo'
    """
    args = "python main_cnn.py --run_mode=test --output_dim={} --evaluate_input_folder={} --evaluate_output_folder={}".format(output_dim, input_folder, output_folder)
    subprocess.call(args, shell=True)


