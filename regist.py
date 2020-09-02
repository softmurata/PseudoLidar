import tensorflow as tf
import copy
import numpy as np
import os
import subprocess
import open3d as o3d

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



def create_point_cloud_from_rgbd(source_rgb_path, source_depth_path, target_rgb_path, target_depth_path, voxel_size, config):
    # read image
    source_rgb = o3d.io.read_image(source_rgb_path)
    source_depth = o3d.io.read_image(source_depth_path)
    target_rgb = o3d.io.read_image(target_rgb_path)
    target_depth = o3d.io.read_image(target_depth_path)

    save_dir = './MonoSmnetData/raw_point_cloud/'

    if not os.path.exists(save_dir):
    	os.makedirs(save_dir)

    # set camera intrinsic
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(config.width, config.height, config.fx, config.fy, config.cx, config.cy)

    # create point cloud
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_rgb, source_depth)
    source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(target_rgb, target_depth)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, pinhole_camera_intrinsic)
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))


    source_pcd_path = '{}source.ply'.format(save_dir)
    target_pcd_path = '{}target.ply'.format(save_dir)

    o3d.io.write_point_cloud(source_pcd_path, source_pcd)
    o3d.io.write_point_cloud(target_pcd_path, target_pcd)

    point_cloud_files = [source_pcd_path, target_pcd_path]

    return point_cloud_files


def prepare_dataset(point_cloud, voxel_size):
	keypoints = point_cloud.voxel_down_sample(voxel_size)
	radius_normal = voxel_size * 2
	# estimate normals line
	keypoints.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

	# calculate fpfh
	radius_feature = voxel_size * 5
	pcd_fpfh = o3d.registration.compute_fpfh_feature(keypoints, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))

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


def save_down_sample_and_features(point_cloud_files, voxel_size):
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

	ds_save_dir = './MonoSmnetData/down_sample/'
	desc_save_dir = './data/demo/32_dim/'
	if not os.path.exists(ds_save_dir):
		os.makedirs(ds_save_dir)

	if not os.path.exists(desc_save_dir):
		os.makedirs(desc_save_dir)


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


def inference(input_folder):
	args = "python main_cnn.py --run_mode=test --evaluate_input_folder={} --evaluate_output_folder=./data/demo".format(input_folder)
	subprocess.call(args, shell=True)


def test():
    voxel_size = 0.01
    input_folder = "./data/demo/test/"
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)

    root_dir = '../rgbd_dataset/RGBD/'
    dataset_name = 'Pikachu'
    dataset_dir = root_dir + dataset_name + '/'
    
    source_fnum = 20
    target_fnum = 22

    source_rgb_path = dataset_dir + 'rgb/{}.png'.format(source_fnum)
    source_depth_path = dataset_dir + 'better_depth/{}.png'.format(source_fnum)

    target_rgb_path = dataset_dir + 'rgb/{}.png'.format(target_fnum)
    target_depth_path = dataset_dir + 'better_depth/{}.png'.format(target_fnum)

    config = Config()


    point_cloud_files = create_point_cloud_from_rgbd(source_rgb_path, source_depth_path, target_rgb_path, target_depth_path, voxel_size, config)

    down_point_cloud_files, desc_files = save_down_sample_and_features(point_cloud_files, voxel_size)

    create_down_pcd(down_point_cloud_files, voxel_size, input_folder)

    inference(input_folder)

    # get descriptors
    source_desc = np.load(desc_files[0])
    source_desc = source_desc['arr_0']
    print(source_desc.shape)

    target_desc = np.load(desc_files[1])
    target_desc = target_desc['arr_0']

    # open3d feature
    source = o3d.registration.Feature()
    source.data = source_desc
    target = o3d.registration.Feature()
    target.data = target_desc

    # Load point cloud
    source_pc = o3d.io.read_point_cloud(point_cloud_files[0])
    target_pc = o3d.io.read_point_cloud(point_cloud_files[1])

    # load point cloud with keypoints
    source_pc_kps = o3d.io.read_point_cloud(down_point_cloud_files[0])
    target_pc_kps = o3d.io.read_point_cloud(down_point_cloud_files[1])
    result_ransac = execute_global_registration(source_pc_kps, target_pc_kps, source, target, 0.05)

    source_temp = copy.deepcopy(source_pc)
    target_temp = copy.deepcopy(target_pc)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    # transformation = result_ransac.transformation
    # source_temp.transform(transformation)

	
    # Iterative Closest Point
    result_icp = refine_registration(source_pc, target_pc, source_pc_kps, target_pc_kps, voxel_size, result_ransac)
        
    source_temp = copy.deepcopy(source_pc)
    target_temp = copy.deepcopy(target_pc)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    transformation = result_icp.transformation
    source_temp.transform(transformation)
        
        
    result_dir = './MonoSmnetData/result/'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
            
    pcd = o3d.geometry.PointCloud()
    source_points = np.asarray(source_temp.points)
    target_points = np.asarray(target_temp.points)

    source_colors = np.asarray(source_temp.colors)
    target_colors = np.asarray(target_temp.colors)

    points = np.concatenate([source_points, target_points], axis=0)
    colors = np.concatenate([source_colors, target_colors], axis=0)

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(result_dir + 'ref.ply', source_temp)
    o3d.io.write_point_cloud(result_dir + 'test.ply', target_temp)
    o3d.io.write_point_cloud(result_dir + 'ans.ply', pcd)




if __name__ == '__main__':
    test()





