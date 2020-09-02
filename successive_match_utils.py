import numpy as np
import open3d as o3d
import subprocess
import os
import cv2
from skimage import io

# camera configuration
class CameraConfig(object):

    def __init__(self):
        self.width = 640
        self.height = 480
        self.fx = 381.694
        self.fy = 381.694
        self.cx = 323.56
        self.cy = 237.11
        self.bx = 0
        self.by = 0



def prepare_materials(rgb_path, depth_path, detection_path, pinhole_camera_intrinsic, args):
    rgb_image = io.imread(rgb_path)
    depth_image = io.imread(depth_path)
    detection_coords = np.load(detection_path)
    
    xmin, ymin, xmax, ymax = detection_coords.astype(int)
    
    detect_rgb_image = np.zeros_like(rgb_image)
    
    detect_rgb_image[ymin:ymax, xmin:xmax] = rgb_image[ymin:ymax, xmin:xmax]
    
    detect_depth_image = np.zeros_like(depth_image)
    detect_depth_image[ymin:ymax, xmin:xmax] = depth_image[ymin:ymax, xmin:xmax]
    
    detect_rgb_dir = args.dataset_dir + args.dataset_name + '/detect_rgb/'
    detect_depth_dir = args.dataset_dir + args.dataset_name + '/detect_depth/'
    
    os.makedirs(detect_rgb_dir, exist_ok=True)
    os.makedirs(detect_depth_dir, exist_ok=True)
    
    fnum = rgb_path.split('/')[-1].split('.')[0]
    
    cv2.imwrite(detect_rgb_dir + fnum + '.png', detect_rgb_image)
    cv2.imwrite(detect_depth_dir + fnum + '.png', detect_depth_image)
    
    # for open3d format
    rgb = o3d.io.read_image(detect_rgb_dir + fnum + '.png')
    depth = o3d.io.read_image(detect_depth_dir + fnum + '.png')
    
    print(type(rgb))
    # create source point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2, max_nn=30))
    
    # calculate keypoints
    keypoints = point_cloud.voxel_down_sample(args.voxel_size)
    radius_normal = args.voxel_size * 2
    keypoints.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # calculate fpfh
    radius_feature = args.voxel_size * 5
    pcd_fpfh = o3d.registration.compute_fpfh_feature(keypoints, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))    
    
    return point_cloud, keypoints, pcd_fpfh, detection_coords


def save_down_sample_and_features(point_clouds, down_point_clouds, features, ds_save_dir, desc_save_dir):
    source_pcd, target_pcd = point_clouds
    source_down, target_down = down_point_clouds
    source_fpfh, target_fpfh = features
    
    source_features = np.asarray(source_fpfh.data)
    target_features = np.asarray(target_fpfh.data)
    
    # down sampling point cloud
    source_down_sample_pcd_path = ds_save_dir + 'source.ply'
    target_down_sample_pcd_path = ds_save_dir + 'target.ply'
    
    o3d.io.write_point_cloud(source_down_sample_pcd_path, source_down)
    o3d.io.write_point_cloud(target_down_sample_pcd_path, target_down)
    
    down_point_cloud_files = [source_down_sample_pcd_path, target_down_sample_pcd_path]
    
    # fpfh
    source_desc_path = desc_save_dir + 'source.npz'
    target_desc_path = desc_save_dir + 'target.npz'
    
    np.savez(source_desc_path, source_features)
    np.savez(target_desc_path, target_features)
    
    desc_files = [source_desc_path, target_desc_path]
    
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

# RANSAC
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

#ICP
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(source, target, distance_threshold,
            result_ransac.transformation,
            o3d.registration.TransformationEstimationPointToPlane())
    return result
    
    


