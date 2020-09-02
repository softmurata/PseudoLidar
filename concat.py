import argparse
import sys
sys.path.append('Capture3D')
from pcd_regist import regist_pcd_with_3dmatch
from clean_raw_point_cloud import clean_pcd

# need files for running 3d match
# global_regist_utils.py
# match3d_utils.py
# clean_raw_point_cloud.py
# pcd_regist.py
# concat.py


def main():
    """
    command line
    python concat.py --voxel_size 0.1 --evaluate_folder ./SmnetData/ --dataset_dir ../rgbd_dataset/RGBD/ --dataset_name Pikachu --output_dim 32 --maxdepth 500 --clean False --finish 20
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--evaluate_folder', type=str, default='./SmnetData/')
    parser.add_argument('--dataset_dir', type=str, default='../rgbd_dataset/RGBD/')
    parser.add_argument('--dataset_name', type=str, default='Pikachu')
    parser.add_argument('--output_dim', type=int, default=32, help='16, 32, 64, 128(but pretrained 128 dim must be installed from github')
    parser.add_argument('--maxdepth', type=int, default=500)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--finish', type=int, default=20)
    args = parser.parse_args()
    
    clean = False
    if clean:
        clean_pcd(args)
    regist_pcd_with_3dmatch(args)        
    
    
if __name__ == '__main__':
    
    main()
