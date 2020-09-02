import os
import glob
import copy
import numpy as np
import open3d as o3d
import sys
sys.path.append('./Capture3D')
from match3d_utils import *

dataset_dir = './SmnetData/Nami7/result_point_cloud/'
output_dir = './SmnetData/Nami7/connect'

os.makedirs(output_dir, exist_ok=True)

output_dim = 32
voxel_size = 0.01
max_nn = 30

pcd_files = glob.glob(dataset_dir + '*.ply')
numbers = [int(d.split('/')[-1].split('.')[0][6:]) for d in pcd_files]
    
sorted_index = np.argsort(numbers)
pcd_files = [pcd_files[idx] for idx in sorted_index]
numbers = [numbers[idx] for idx in sorted_index]


# source point cloud
source_pcd = o3d.io.read_point_cloud(pcd_files[0])
source_pcd_dir = output_dir + '/' + 'result_point_cloud/'

os.makedirs(source_pcd_dir, exist_ok=True)

source_pcd_path = source_pcd_dir + 'source.ply'
o3d.io.write_point_cloud(source_pcd_path, source_pcd)

source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=max_nn))

pcd_files = pcd_files[1:]
numbers = numbers[1:]

ds_save_dir = './SmnetData/Nami7/connect/down_sample/'
desc_save_dir = './SmnetData/Nami7/connect/{}_dim/'.format(output_dim)

os.makedirs(ds_save_dir, exist_ok=True)
os.makedirs(desc_save_dir, exist_ok=True)    

for fnum, target_pcd_path in zip(numbers, pcd_files):
    point_cloud_files = [source_pcd_path, target_pcd_path]
    # down sampling
    target_pcd = o3d.io.read_point_cloud(target_pcd_path)
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
    
    # down sample
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
    
    # generate point cloud from down sampling
    create_down_pcd(down_point_cloud_files, voxel_size, output_dir)
    
    # inference
    inference(output_dim, output_dir + '/', output_dir + '/')
    
    # get descriptors
    source_desc = np.load(desc_files[0])
    source_desc = source_desc['arr_0']
        
    target_desc = np.load(desc_files[1])
    target_desc = target_desc['arr_0']
        
    # convert descriptor into open3d Feature class
    source_f = o3d.registration.Feature()
    source_f.data = source_desc
        
    target_f = o3d.registration.Feature()
    target_f.data = target_desc
        
    # load point cloud
    source_pc = o3d.io.read_point_cloud(point_cloud_files[0])
    target_pc = o3d.io.read_point_cloud(point_cloud_files[1])
    source_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=max_nn))
    target_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=max_nn))
    
        
    # load point cloud with keypoints
    source_pc_kps = o3d.io.read_point_cloud(down_point_cloud_files[0])
    target_pc_kps = o3d.io.read_point_cloud(down_point_cloud_files[1])
        
    # ransac
    result_ransac = execute_global_registration(source_pc_kps, target_pc_kps, source_f, target_f, voxel_size)
        
    # icp
    result_icp = refine_registration(source_pc, target_pc, source_f, target_f, voxel_size, result_ransac)
        
    # get transfomation matrix
    transformation = result_icp.transformation
    
    source_temp = copy.deepcopy(source_pc)
    target_temp = copy.deepcopy(target_pc)
        
    source_temp.transform(transformation)
        
    # create new source point cloud
    # but memory over
    points = np.asarray(source_temp.points).tolist()
    colors = np.asarray(source_temp.colors).tolist()
        
    points.extend(np.asarray(target_temp.points).tolist())
    colors.extend(np.asarray(target_temp.colors).tolist())
        
    # create new source point cloud(In fact, concatenate with previous source point cloud and new one)
    source_pcd = o3d.geometry.PointCloud()

    tar_pt = np.array(points)
    tar_col = np.array(colors)

    print()
    print('----confirmation-----')
    print(tar_pt.shape)
    print()
    source_pcd.points = o3d.utility.Vector3dVector(tar_pt)
    source_pcd.colors = o3d.utility.Vector3dVector(tar_col)
    # source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2, max_nn=max_nn))
        
    del points
    del colors
    del tar_pt
    del tar_col

    print('source pcd finish?')
    # save source point cloud
    o3d.io.write_point_cloud(source_pcd_path, source_pcd)
    

