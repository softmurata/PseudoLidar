import numpy as np
import open3d as o3d

path = '/mnt/c/Users/TeamvRAN/Downloads/001300.npy'

arr = np.load(path)

print(arr)
print(arr.shape)

colors = np.random.rand(arr.shape[0], arr.shape[1])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(arr)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud('lidar.ply', pcd)

