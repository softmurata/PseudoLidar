import glob
import argparse
import numpy as np
import os
import open3d as o3d
from successive_match_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='../rgbd_dataset/RGBD/')
parser.add_argument('--dataset_name', type=str, default='Nami7')
parser.add_argument('--output_dim', type=int, default=32)
parser.add_argument('--interval', type=int, default=3)
parser.add_argument('--voxel_size', type=float, default=0.01)
args = parser.parse_args()

# successive match

# extract object by DETR

# Input directory
dataset_name = args.dataset_name  # dataset_name = args.dataset_name
dataset_dir = args.dataset_dir + dataset_name + '/'

depth_dir = dataset_dir + 'better_depth/'
detection_dir = dataset_dir + 'detection/'  # files which has information about coords of object
rgb_dir = dataset_dir + 'rgb/'
calib_dir = dataset_dir + 'calib/'

os.makedirs(calib_dir, exist_ok=True)

rgb_files = glob.glob(rgb_dir + '*.png')
numbers = [int(r.split('/')[-1].split('.')[0]) for r in rgb_files]

sorted_index = np.argsort(numbers)
rgb_files = [rgb_files[idx] for idx in sorted_index]
depth_files = [depth_dir + str(numbers[idx]) + '.png' for idx in sorted_index]
detection_files = [detection_dir + str(numbers[idx]) + '.npy' for idx in sorted_index]

numbers = [numbers[idx] for idx in sorted_index]

# Output directory
output_dir = './SucSmnetData/{}'.format(dataset_name)
os.makedirs(output_dir, exist_ok=True)

# file path to restore descriptors and down sample point cloud
ds_save_dir = output_dir + '/down_sample/'
desc_save_dir = output_dir + '/{}_dim/'.format(args.output_dim)

os.makedirs(ds_save_dir, exist_ok=True)
os.makedirs(desc_save_dir, exist_ok=True)


# set camera configuration
config = CameraConfig()

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(config.width, config.height, config.fx, config.fy, config.cx, config.cy)



for fidx in range(len(numbers) - 1):
    
    # set data path
    # source
    source_rgb_path = rgb_files[fidx]
    source_depth_path = depth_files[fidx]
    source_detection_path = detection_files[fidx]
    
    # target
    target_rgb_path = rgb_files[fidx + 1]
    target_depth_path = depth_files[fidx + 1]
    target_detection_path = detection_files[fidx + 1]
    
    # prepare dataset
    source_point_cloud, source_down, source_fpfh, source_detection_coords = prepare_materials(source_rgb_path, source_depth_path, source_detection_path, pinhole_camera_intrinsic, args)
    target_point_cloud, target_down, target_fpfh, target_detection_coords = prepare_materials(target_rgb_path, target_depth_path, target_detection_path, pinhole_camera_intrinsic, args)
    
    # save information
    down_point_cloud_files, desc_files = save_down_sample_and_features([source_point_cloud, target_point_cloud], [source_down, target_down], [source_fpfh, target_fpfh], ds_save_dir, desc_save_dir)
    # generate point cloud from down sampling
    create_down_pcd(down_point_cloud_files, args.voxel_size, output_dir)
    # inference
    inference(args.output_dim, output_dir + '/', output_dir + '/')
    
    # ransac
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, args.voxel_size)
    
    # ICP
    result_icp = refine_registration(source_point_cloud, target_point_cloud, source_fpfh, target_fpfh, args.voxel_size, result_ransac)
    
    # get rotation matrix
    transformation = result_icp.transformation
    
    np.save(calib_dir + '{}.npy'.format(numbers[fidx + 1]), transformation)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
