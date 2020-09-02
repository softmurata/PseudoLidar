from __future__ import print_function

import numpy as np
import yaml


class RealsenseCalibration(object):
    
    def __init__(self, realsense_intrinsic, realsense_calib_path='./Mydataset/LIDAR/calib/0.npy'):
        # get intrinsic parameters
        file = open(realsense_intrinsic, 'r+')
        data = yaml.load(file)
        # get pose
        # pose = np.load(realsense_calib_path)
        # self.R = pose[:, :3]
        # self.t = pose[:, 3:]
        self.R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.t = np.array([[0, 0, 0]], dtype=np.float32)
        # from kitti dataset
        V2C = np.array([[7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03],
                        [1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02]
                        [9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01]])
        
        # or calculate the rotation matrix of image center point and velodyne center point from optimization
        
        # C2V
        inv_tr = np.zeros_like(self.V2C)
        inv_tr[0:3, 0:3] = np.transpose(self.V2C[0:3, 0:3])
        inv_tr[0:3, 3] = np.dot(-np.transpose(self.V2C[0:3, 0:3]), self.V2C[0:3, 3])
        self.C2V = inv_tr
        
        """
        # 
        C2V = [[ 0.  0.  1.  0.]
        [-1.  0.  0.  0.]
        [ 0. -1.  0.  0.]]
        """
        
        self.P = np.array([[data['fx'], 0,          data['cx'], 0],
                           [0,          data['fy'], data['cy'], 0],
                           [0,          0,          1,          0]])
        
        self.cu = data['cx']  # x center coord
        self.cv = data['cy']  # y center coord
        self.fu = data['fx']  # x focal length
        self.fv = data['fy']  # y focal length
        self.bx = data['bx']  # x baseline corresponding to reference(basically 0)
        self.by = data['by']  # same as bx
        
    def cart2hom(self, pts_3d):
        """
        Input: n * 3
        Output: n * 4
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom
    
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)
        return np.dot(pts_3d_velo, np.transpose(self.V2C))
    
    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)
        return np.dot(pts_3d_ref, np.transpose(self.C2V))
    
    def project_rect_to_ref(self, pts_3d_rect):
        
        return np.transpose(np.dot(np.linalg.inv(self.R), np.transpose(pts_3d_rect)))
    
    def project_ref_to_rect(self, pts_3d_ref):
        
        return np.transpose(np.dot(self.R, np.transpose(pts_3d_ref)))
    
    def project_rect_to_velo(self, pts_3d_rect):
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        velodyne = self.project_ref_to_velo(pts_3d_ref)
        
        return velodyne
    
    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        pts_3d_rect = self.project_ref_to_rect(pts_3d_ref)
        return pts_3d_rect
    
    # 3D => 2D
    def project_rect_to_image(self, pts_3d_rect):
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]
    
    def project_velo_to_image(self, pts_3d_velo):
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        image = self.project_rect_to_image(pts_3d_rect)
        return image
    
    # 2D => 3D
    def project_image_to_rect(self, uv_depth):
        """
        Input: n * 3, first channels uv, 3rd channels depth
        output: n * 3 points in rect camera coord
        """
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.cu) * uv_depth[:, 2]) / self.fu + self.bx
        y = ((uv_depth[:, 1] - self.cv) * uv_depth[:, 2]) / self.fv + self.by
        
        # Initialize 3d points
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        
        return pts_3d_rect
        
    def project_rect_to_velo(self, pts_3d_rect):
        """
        Input: n * 3 points in rect camera coord
        Output: n * 3 points in velodyne coords
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        velodyne = self.project_ref_to_velo(pts_3d_ref)
        
        return velodyne
        
    
    
    # image to velodyne(point cloud)
    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        velodyne = self.project_rect_to_velo(pts_3d_rect)
        
        return velodyne
    
    
    

# kitti utils code
# Calibration class
class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath):

        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

        
        
            
        
        
        


if __name__ == '__main__':
    realsense_intrinsic = 'realsense.yaml'
    realsense_calib_path = 'calib/5.npy'  # R0 data
    rsc = RealsenseCalibration(realsense_intrinsic, realsense_calib_path)


