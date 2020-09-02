import open3d as o3d
import argparse
import os
import glob
import copy
from clean_raw_point_cloud import clean_pcd
from match3d_utils import *

    
def regist_pcd_with_3dmatch(args):
    
    input_folder = args.evaluate_folder + args.dataset_name  # 'SmnetData/Pikachu'
    output_folder = input_folder
    
    os.makedirs(input_folder, exist_ok=True)
    
    max_nn = 30

    input_pcd_dir = input_folder + '/' + 'pre_point_cloud/'
    pcd_files = glob.glob(input_pcd_dir + '*.ply')
    numbers = [int(d.split('/')[-1].split('.')[0][6:]) for d in pcd_files]
    
    sorted_index = np.argsort(numbers)
    pcd_files = [pcd_files[idx] for idx in sorted_index][args.start:args.finish]
    numbers = [numbers[idx] for idx in sorted_index][args.start:args.finish]
    
    # for save
    new_pcd_dir = input_folder + '/' + 'result_point_cloud'
    new_calib_dir = input_folder + '/' + 'calib'
    
    os.makedirs(new_pcd_dir, exist_ok=True)
    os.makedirs(new_calib_dir, exist_ok=True)
    
    # Initialize source point cloud
    source_pcd = o3d.io.read_point_cloud(pcd_files[0])
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2, max_nn=max_nn))
    
    source_pcd_path = new_pcd_dir + '/' + 'source{}.ply'.format(args.divide_num)
    o3d.io.write_point_cloud(source_pcd_path, source_pcd)
    
    pcd_files = pcd_files[1:]
    numbers = numbers[1:]
    
    for fnum, target_pcd_path in zip(numbers, pcd_files):
        if fnum % 3 != 2:
            continue
        point_cloud_files = [source_pcd_path, target_pcd_path]
        print()
        print('---path---')
        print(point_cloud_files)
        print()
        # down sampling
        down_point_cloud_files, desc_files = save_down_sample_and_features(point_cloud_files, args)
        # generate point cloud from down sampling
        create_down_pcd(down_point_cloud_files, args.voxel_size, input_folder)
        # inference
        inference(args.output_dim, input_folder + '/', output_folder + '/')
        
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
        source_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2, max_nn=max_nn))
        target_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2, max_nn=max_nn))
    
        
        # load point cloud with keypoints
        source_pc_kps = o3d.io.read_point_cloud(down_point_cloud_files[0])
        target_pc_kps = o3d.io.read_point_cloud(down_point_cloud_files[1])
        
        # ransac
        result_ransac = execute_global_registration(source_pc_kps, target_pc_kps, source_f, target_f, args.voxel_size)
        
        # icp
        result_icp = refine_registration(source_pc, target_pc, source_f, target_f, args.voxel_size, result_ransac)
        
        # get transfomation matrix
        transformation = result_icp.transformation
        
        np.save(new_calib_dir + '/' + 'target{}.npy'.format(fnum), transformation)
        
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
    
    

