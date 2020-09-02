import open3d as o3d
import numpy as np

# utils
# estinate normal vector and features
def preprocess_point_cloud(point_cloud, voxel_size):
    # get keypoints by voxel downsampling
    pcd_down = point_cloud.voxel_down_sample(voxel_size)
    
    # estimate normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    # view_point = np.array([0, 10, 10], dtype="float64")
    # pcd_down.orient_normals_toward_camera_location(camera_location=view_point)
    
    # calculate fpfh features
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=150))
    
    return pcd_down, pcd_fpfh

# Ransac
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5  # 2.5?
    # RANSAC
    print('----RANSAC part-----')
    result = o3d.registration.registration_ransac_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
                                                                            o3d.registration.TransformationEstimationPointToPoint(False), 4,
                                                                            [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
                                                                             o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                                                                            o3d.registration.RANSACConvergenceCriteria(5000000, 1000))
    
    return result

# ICP refinement(LOCAL refinement)
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print('----ICP part-----')
    result = o3d.registration.registration_icp(source, target, distance_threshold, result_ransac.transformation,
                                               o3d.registration.TransformationEstimationPointToPlane(),
                                               o3d.registration.ICPConvergenceCriteria(max_iteration=50))
    
    return result

# get oriented bounting box and axis aligned box
def get_bbox_and_aabox(pcd):
    # source center(bbox.get_center())   
    base_center = np.array([ 0.16436595, -0.02519854, 1.43643664])  # 10.png
    # Oriented Bounting Box
    points = np.asanyarray(pcd.points)
    colors = np.asanyarray(pcd.colors)
        
    # create oriented bounting box
    oribox = o3d.geometry.OrientedBoundingBox()
    bbox = oribox.create_from_points(o3d.utility.Vector3dVector(points))
        
    print()
    print('--orientation bbox info--')
    print('center:', bbox.get_center())

    current_center = bbox.get_center()
    center_move = current_center - base_center
    points -= center_move

    print('--edge 8 points--')
    bbox_edges = np.asarray(bbox.get_box_points())
    print(bbox_edges)
        
    # create axis aligned bbox
    axalbox = o3d.geometry.AxisAlignedBoundingBox()
    aabox = axalbox.create_from_points(o3d.utility.Vector3dVector(points))
        
    print()
    print('--axis aligned bbox info--')
    print('center:', aabox.get_center())
    print('8 edges:', aabox.get_box_points())

    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

