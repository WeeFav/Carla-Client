import carla
import numpy as np   
import config as cfg 
import math
import time
import cv2


class BEV():
    def __init__(self, camera_spawnpoint):
        self.c_x = cfg.image_width / 2
        self.c_y = cfg.image_height / 2
        self.f = cfg.image_width / (2 * math.tan(cfg.fov * math.pi / 360))

        self.camera_extrinsic = self.get_camera_extrinsic_matrix(camera_spawnpoint)

        # BEV area in meters (vehicle-centered)
        self.x_range = (-10, 40)  # From 10 meters behind to 40 meters front
        self.y_range = (-20, 20)  # From 20 meters left to 20 meters right
        self.resolution = 0.1     # meters per pixel (i.e., 10 pixels/m)

        # Image size
        self.bev_width  = int((self.y_range[1] - self.y_range[0]) / self.resolution)   # columns
        self.bev_height = int((self.x_range[1] - self.x_range[0]) / self.resolution)   # rows


    def get_camera_extrinsic_matrix(self, camera_spawnpoint):
        rotation = camera_spawnpoint.rotation
        location = camera_spawnpoint.location

        # Convert CARLA rotation (in degrees) to radians and then to matrix
        pitch = math.radians(rotation.pitch)
        yaw   = math.radians(rotation.yaw)
        roll  = math.radians(rotation.roll)

        # Create rotation matrix from pitch, yaw, roll (extrinsic to camera)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx
        T = np.array([location.x, location.y, location.z]).reshape((3, 1))

        # Combine into 4x4 matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T.flatten()

        return extrinsic


    # 2D pixel to 3D camera coordinates 
    def pixel_to_camera(self, points_2d, image_depth):
        x_coords = points_2d[:, 0].astype(int) # (N,)
        y_coords = points_2d[:, 1].astype(int)

        rgb = image_depth[y_coords, x_coords] # (N, 3)
        R = rgb[:, 0].astype(int) # (N,)
        G = rgb[:, 1].astype(int)
        B = rgb[:, 2].astype(int)

        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        depth_in_meters = 1000 * normalized

        x_camera = (x_coords - self.c_x) * depth_in_meters / self.f
        y_camera = (y_coords - self.c_y) * depth_in_meters / self.f
        z_camera = depth_in_meters

        points_camera = np.stack((x_camera, y_camera, z_camera, np.ones_like(x_camera)), axis=0).T

        return points_camera


    def camera_to_vehicle(self, points_camera):
        camera_to_vehicle = np.linalg.inv(self.camera_extrinsic)
        points_vehicle = camera_to_vehicle @ points_camera.T
        return points_vehicle


    def vehicle_to_bev(self, x, y):
        u = int((x - self.x_range[0]) / self.resolution)
        v = int((self.y_range[1] - y) / self.resolution)
        # u = int((y - self.y_range[0]) / self.resolution)  # left to right becomes u=0 to width
        # v = int((self.x_range[1] - x) / self.resolution)  # front to back becomes v=0 to height
        return u, v


    def get_bev_view(self, points_2d, image_depth):
        bev_image = np.zeros((self.bev_height, self.bev_width), dtype=np.uint8)

        points_camera = self.pixel_to_camera(points_2d, image_depth)
        points_vehicle = self.camera_to_vehicle(points_camera)

        for x, y, _, _ in points_vehicle.T:  # points_vehicle: list of (X, Y) in vehicle frame
            if self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]:
                u, v = self.vehicle_to_bev(x, y)
                bev_image[v, u] = 255  # mark pixel

        bev_color = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
        cv2.imshow("BEV", bev_color)
        cv2.waitKey(1)

