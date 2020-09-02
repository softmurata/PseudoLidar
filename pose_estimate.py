import open3d as o3d
import numpy as np
import os

# ToDo: if you use 3D matching, you improve the quality of point cloud
# ToDo: next step => with pseudo lidar
def compose_point_cloud(args):
    
    source_pcd_path = './PCD/{}/point_cloud/target1.ply'.format(args.dataset_name)
    R = np.load('./PCD/{}/calib/target1.npy'.format(args.dataset_name))
    
    source_pcd = o3d.io.read_point_cloud(source_pcd_path)
    
    num_files = os.listdir('./PCD/{}/calib/'.format(args.dataset_name))
    indicies = [int(c.split('.')[0][6:]) for c in num_files]
    sorted_index = np.argsort(indicies)
    calib_files = ['./PCD/{}/calib/'.format(args.dataset_name) + num_files[idx] for idx in sorted_index][:args.finish][1:]
    pcd_files = ['./PCD/{}/point_cloud/'.format(args.dataset_name) + num_files[idx].split('.')[0] + '.ply' for idx in sorted_index][:args.finish][1:]
    
    # list of rotation matrix list
    Rs = []
    
    for calib_path in calib_files:
        R_now = np.load(calib_path)
        R = np.dot(R, R_now)
        Rs.append(R)
        
    Rs = Rs[::-1]  # reverse
    
    # list of point and colors
    total_points = []
    total_colors = []
    for idx, pcd_path in enumerate(pcd_files):
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd.transform(Rs[idx])
        total_points.extend(np.asarray(pcd.points))
        total_colors.extend(np.asarray(pcd.colors))
        
    total_points = np.array(total_points)
    total_colors = np.array(total_colors)
    
    ans_pcd = o3d.geometry.PointCloud()
    ans_pcd.points = o3d.utility.Vector3dVector(total_points)
    ans_pcd.colors = o3d.utility.Vector3dVector(total_colors)
    
    o3d.io.write_point_cloud('result.ply', ans_pcd)








