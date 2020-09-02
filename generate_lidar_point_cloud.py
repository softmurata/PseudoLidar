from __future__ import print_function
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from skimage import io
import cv2
import open3d as o3d
from calibration import RealsenseCalibration, Calibration

# TestImageLoader
# predict_disparity from scene model flow
# save disparity


def project_disp_to_depth(calib, rgb, depth, min_high, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    print('points shape:', points.shape)
    rgb = rgb.transpose(2, 0, 1) / 255  # transpose and normalize
    colors = rgb.reshape((3, -1))
    points = points.reshape((3, -1))
    
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= min_high) & (cloud[:, 2] < max_high)
    
    true_colors = colors.T[valid]
    true_cloud = cloud[valid]
    
    # put together color and point
    # pcd_with_colors = np.concatenate([true_cloud, true_colors], axis=0)
    
    return true_cloud, true_colors

# calibration matrix derives from mask fusion slam
def run_with_mask_fusion_calib():
    parser = argparse.ArgumentParser(description='Generate Lidar Point Cloud')
    parser.add_argument('--dataset', type=str, default='./MyDataset/')
    parser.add_argument('--fnum', type=int, default=130)
    parser.add_argument('--realsense_intrinsic', type=str, default='realsense.yaml')
    parser.add_argument('--save_dir', type=str, default='./own_results/')
    parser.add_argument('--save_fname', type=str, default='test')
    parser.add_argument('--min_high', type=float, default=0)
    parser.add_argument('--max_high', type=float, default=1)
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    depth_file = args.dataset + 'depth/{}.npy'.format(args.fnum)
    rgb_file = args.dataset + 'rgb/{}.png'.format(args.fnum)
    realsense_calib_path = args.dataset + 'calib/{}.npy'.format(args.fnum)
    
    # load file
    calib = RealsenseCalibration(args.realsense_intrinsic, realsense_calib_path)
    rgb = io.imread(rgb_file)
    depth = np.load(depth_file)
    
    
    # generate lidar point cloud with colors
    lidar_pcd, lidar_pcd_colors = project_disp_to_depth(calib, rgb, depth, args.min_high, args.max_high)
    
    print(lidar_pcd.shape)
    
    # save raw_point and point_cloud
    raw_point_save = args.save_dir + 'raw_point/'
    if not os.path.exists(raw_point_save):
        os.mkdir(raw_point_save)
        
    point_cloud_save = args.save_dir + 'point_cloud/'
    if not os.path.exists(point_cloud_save):
        os.mkdir(point_cloud_save)
        
    # save raw point
    np.save(raw_point_save + '{}.npy'.format(args.save_fname), lidar_pcd)
    
    # create point cloud
    pcd = o3d.geometry.PointCloud()
    # restore points and colors in point cloud class
    pcd.points = o3d.utility.Vector3dVector(lidar_pcd)
    pcd.colors = o3d.utility.Vector3dVector(lidar_pcd_colors)
    
    # save point cloud
    o3d.io.write_point_cloud(point_cloud_save + '{}.ply'.format(args.save_fname), pcd)


def run_with_capture_3d_calib():
    parser = argparse.ArgumentParser(description='Generate Lidar')
    parser = argparse.ArgumentParser(description='Generate Lidar Point Cloud')
    parser.add_argument('--dataset', type=str, default='./MyDataset/Data/')
    parser.add_argument('--fnum', type=int, default=130)
    parser.add_argument('--realsense_intrinsic', type=str, default='realsense.yaml')
    parser.add_argument('--save_dir', type=str, default='./own_results/')
    parser.add_argument('--save_fname', type=str, default='test')
    parser.add_argument('--min_high', type=float, default=0)
    parser.add_argument('--max_high', type=float, default=1)
    args = parser.parse_args()
    
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    depth_file = args.dataset + 'depth/{}.npy'.format(args.fnum)
    rgb_file = args.dataset + 'rgb/{}.png'.format(args.fnum)
    
    # load file
    calib = RealsenseCalibration(args.realsense_intrinsic)
    rgb = io.imread(rgb_file)
    depth = np.load(depth_file)
    
    # generate lidar point cloud with colors
    lidar_pcd, lidar_pcd_colors = project_disp_to_depth(calib, rgb, depth, args.min_high, args.max_high)
    
    print(lidar_pcd.shape)
    
    # save raw_point and point_cloud
    raw_point_save = args.save_dir + 'raw_point/'
    if not os.path.exists(raw_point_save):
        os.mkdir(raw_point_save)
        
    point_cloud_save = args.save_dir + 'point_cloud/'
    if not os.path.exists(point_cloud_save):
        os.mkdir(point_cloud_save)
        
    # save raw point
    np.save(raw_point_save + '{}.npy'.format(args.save_fname), lidar_pcd)
    
    # create point cloud
    pcd = o3d.geometry.PointCloud()
    # restore points and colors in point cloud class
    pcd.points = o3d.utility.Vector3dVector(lidar_pcd)
    pcd.colors = o3d.utility.Vector3dVector(lidar_pcd_colors)
    
    # save point cloud
    o3d.io.write_point_cloud(point_cloud_save + '{}.ply'.format(args.save_fname), pcd)
       


# calibration matrix derives from kitti dataset
def run_with_kitti_calib():
    parser = argparse.ArgumentParser(description='Generate Lidar')
    parser.add_argument('--calib_file', type=str,
                        default='./KiTTi/training/calib/001000.txt')  # fixed
    parser.add_argument('--rgb_file', type=str,
                        default='./MyDataset/rgb/test.png')
    parser.add_argument('--depth_file', type=str,
                        default='./MyDataset/depth/test.npy')  # from ReaslSense Depth data
    parser.add_argument('--save_dir', type=str,
                        default='./own_results/')
    parser.add_argument('--save_fname', type=str, default='test')
    parser.add_argument('--min_high', type=float, default=0)
    parser.add_argument('--max_high', type=float, default=1)
    args = parser.parse_args()
    
    # rgb image + depth image
    # results => own_results/raw_point/****.npy, own_results/point_cloud/****.ply
    
    depth_fn = args.depth_file
    calib_file = args.calib_file
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # create calib class
    calib = Calibration(calib_file)
    
    depth_map = np.load(depth_fn)
    
    rgb = io.imread(rgb_file)
    
    lidar_pcd, lidar_pcd_colors = project_disp_to_depth(calib, rgb, depth_map, args.min_high, args.max_high)
    
    print(lidar_pcd.shape)
    
    # save raw_point and point_cloud
    raw_point_save = save_dir + 'raw_point/'
    if not os.path.exists(raw_point_save):
        os.mkdir(raw_point_save)
        
    point_cloud_save = save_dir + 'point_cloud/'
    if not os.path.exists(point_cloud_save):
        os.mkdir(point_cloud_save)
        
    # save raw point
    np.save(raw_point_save + '{}.npy'.format(args.save_fname), lidar_pcd)
    
    # create point cloud
    pcd = o3d.geometry.PointCloud()
    # restore points and colors in point cloud class
    pcd.points = o3d.utility.Vector3dVector(lidar_pcd)
    pcd.colors = o3d.utility.Vector3dVector(lidar_pcd_colors)
    
    # save point cloud
    o3d.io.write_point_cloud(point_cloud_save + '{}.ply'.format(args.save_fname), pcd)
    


# main funtion (1 file)
if __name__ == '__main__':
    run_with_capture_3d_calib()
    # realsense
    # run_with_mask_fusion_calib()
    # kitti
    # run_with_kitti_calib()
        
    
    

