import argparse
import os
import glob
import numpy as np
import cv2
from skimage import io
import open3d as o3d
from camera_config import Config

def create_no_noise_depth_map(args):
    # create dataset directory
    dataset_dir = args.dataset_dir + args.dataset_name + '/'
    rgb_dir = dataset_dir + 'rgb/'
    depth_dir = dataset_dir + 'depth/'
    detection_dir = dataset_dir + 'detection/'
    
    rgb_files = glob.glob(rgb_dir + '*.png')
    depth_files = glob.glob(depth_dir + '*.png')
    detection_files = glob.glob(detection_dir + '*.npy')
    numbers = [int(d.split('/')[-1].split('.')[0]) for d in depth_files]
    sorted_index = np.argsort(numbers)
    rgb_files = [rgb_files[idx] for idx in sorted_index]
    depth_files = [depth_files[idx] for idx in sorted_index]
    detection_files = [detection_files[idx] for idx in sorted_index]
    numbers = [numbers[idx] for idx in sorted_index]
    
    # Initialize new depth directory
    better_depth_dir = dataset_dir + 'better_depth/'

    if not os.path.exists(better_depth_dir):
        os.makedirs(better_depth_dir)
        
    
    new_depth_files = []

    H, W = cv2.imread(rgb_files[0]).shape[:2]

    padding_depth = 5000  # hypartameters, if you want to get better results, you should adjust this value
    
    for fnum, rgb_path, depth_path, detection_path in zip(numbers, rgb_files, depth_files, detection_files):
        # load depth file
        depth = io.imread(depth_path)

        # load bounting box information
        xmin, ymin, xmax, ymax = np.load(detection_path).astype(int)

        # Initialize uv map
        uvs = np.zeros_like(depth)

        uvs[ymin:ymax, xmin:xmax] = depth[ymin:ymax, xmin:xmax]
        uvs[:ymin, :xmin] = padding_depth
        uvs[ymax:, xmax:] = padding_depth
        print()
        print('---before depth value information----')
        print(np.max(uvs), np.min(uvs))
        better = uvs < args.maxdepth
        depth = np.multiply(better, uvs)
        print('---after depth value information---')
        print(np.max(depth))
                
        new_depth_path = better_depth_dir + '{}.png'.format(fnum)
        new_depth_files.append(new_depth_path)
        
        cv2.imwrite(new_depth_path, depth)
        
    print('finish new depth map')
        
    return new_depth_files, rgb_files, numbers
    


def clean_pcd(args):
    # remove noise from depth map
    new_depth_files, rgb_files, numbers = create_no_noise_depth_map(args)
    
    # set camera parameters
    config = Config()

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(config.width, config.height, config.fx, config.fy, config.cx, config.cy)
    
    clean_dir = './SmnetData/{}/pre_point_cloud'.format(args.dataset_name)
    os.makedirs(clean_dir, exist_ok=True)
    
    for fnum, rgb_path, new_depth_path in zip(numbers, rgb_files, new_depth_files):
        rgb = o3d.io.read_image(rgb_path)
        depth = o3d.io.read_image(new_depth_path)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2, max_nn=30))
        
        o3d.io.write_point_cloud(clean_dir + '/' + 'target{}.ply'.format(fnum), point_cloud)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='../rgbd_dataset/RGBD/')
    parser.add_argument('--dataset_name', type=str, default='Pikachu')
    parser.add_argument('--maxdepth', type=float, default=500)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    args = parser.parse_args()
    
    clean_pcd(args)


if __name__ == '__main__':
    main()