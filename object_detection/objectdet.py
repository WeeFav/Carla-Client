import numpy as np
import torch

from pointpillars.model.pointpillars import PointPillars
from . import objectdet_config as cfg

class ObjDet():
    def __init__(self):
        CLASSES = {
            'Car': 0,
            'Large': 1,
            'Motorcycle': 2 
        }
        LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
        self.pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

        self.model = PointPillars(nclasses=len(CLASSES)).cuda()
        self.model.load_state_dict(torch.load(cfg.model_path))
        self.model.eval()


    def point_range_filter(self, pts, point_range=[-70.4, -40.0, -3, 70.4, 40.0, 1]):
        flag_x_low = pts[:, 0] > point_range[0]
        flag_y_low = pts[:, 1] > point_range[1]
        flag_z_low = pts[:, 2] > point_range[2]
        flag_x_high = pts[:, 0] < point_range[3]
        flag_y_high = pts[:, 1] < point_range[4]
        flag_z_high = pts[:, 2] < point_range[5]
        keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
        pts = pts[keep_mask]
        return pts


    def keep_bbox_from_lidar_range(self, result, pcd_limit_range):
        '''
        result: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)

        return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
        '''
        lidar_bboxes, labels, scores = result['lidar_bboxes'], result['labels'], result['scores']
        if 'bboxes2d' not in result:
            result['bboxes2d'] = np.zeros_like(lidar_bboxes[:, :4])
        if 'camera_bboxes' not in result:
            result['camera_bboxes'] = np.zeros_like(lidar_bboxes)
        bboxes2d, camera_bboxes = result['bboxes2d'], result['camera_bboxes']
        flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :] # (n, 3)
        flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :] # (n, 3)
        keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)
        
        result = {
            'lidar_bboxes': lidar_bboxes[keep_flag],
            'labels': labels[keep_flag],
            'scores': scores[keep_flag],
            'bboxes2d': bboxes2d[keep_flag],
            'camera_bboxes': camera_bboxes[keep_flag]
        }
        return result


    def predict(self, pointcloud):
        pc = self.point_range_filter(pointcloud)
        pc_torch = torch.from_numpy(pc)
        
        with torch.no_grad():
            pc_torch = pc_torch.cuda()
            result_filter = self.model(batched_pts=[pc_torch], mode='test')[0]
            """
            result_filter = {
                bboxes: [(k1, 7), (k2, 7), ... ] (N, 7)
                labels: [(k1, ), (k2, ), ... ] (N,)
                scores: [(k1, ), (k2, ), ... ] (N,)
            } 
            """
        
        # result_filter = self.keep_bbox_from_lidar_range(result_filter, self.pcd_limit_range)
        if result_filter is not None:
            lidar_bboxes = result_filter['lidar_bboxes'] # (N, 7)
            lidar_bboxes = lidar_bboxes[:, [0, 1, 2, 5, 3, 4, 6]] # wlh -> hwl
            
            labels, scores = result_filter['labels'], result_filter['scores'] # (N,)
        else:
            lidar_bboxes = []
            labels = []

        return lidar_bboxes, labels