import carla
import random
import cv2
import numpy as np
import sys
import pygame
import os
import argparse
from PIL import Image
import time
import math
import open3d as o3d

import config as cfg
from carla_sync_mode import CarlaSyncMode
from vehicle_manager import VehicleManager
from lane_detection.lanemarkings import LaneMarkings
from lane_detection.lanedet import LaneDet
from object_detection.objectdet import ObjDet
from object_detection import gt_bbox
from controllers.pure_pursuit import PurePursuit
from controllers.pid import PID
from utils import bbox3d2corners

class CarlaGame():
    def __init__(self):
        self.display = pygame.display.set_mode((cfg.image_width, cfg.image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 36) 

        self.client = carla.Client('localhost', 2000)
        self.world = self.client.load_world(cfg.town)
        self.map = self.world.get_map()
        self.world.set_weather(cfg.weather)
        self.tm = self.client.get_trafficmanager()

        self.vehicle_manager = VehicleManager(self.client, self.world, self.tm)
        self.ego_vehicle = self.vehicle_manager.spawn_ego_vehicle(autopilot=cfg.carla_auto_pilot)

        blueprint_library = self.world.get_blueprint_library()
        # Spawn rgb-cam and attach to vehicle
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_rgb.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_rgb.set_attribute('fov', f'{cfg.fov}')
        self.camera_spawnpoint = carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=-18.5)) # camera 5
        self.camera_rgb = self.world.spawn_actor(bp_camera_rgb, self.camera_spawnpoint, attach_to=self.ego_vehicle)

        if not cfg.predict_lane:
            # Spawn semseg-cam and attach to vehicle
            bp_camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
            bp_camera_semseg.set_attribute('image_size_x', f'{cfg.image_width}')
            bp_camera_semseg.set_attribute('image_size_y', f'{cfg.image_height}')
            bp_camera_semseg.set_attribute('fov', f'{cfg.fov}')
            self.camera_semseg = self.world.spawn_actor(bp_camera_semseg, self.camera_spawnpoint, attach_to=self.ego_vehicle)
        else:
            self.camera_semseg = None

        # Spawn depth-cam and attach to vehicle
        bp_camera_depth = blueprint_library.find('sensor.camera.depth')
        bp_camera_depth.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_depth.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_depth.set_attribute('fov', f'{cfg.fov}')
        self.camera_depth = self.world.spawn_actor(bp_camera_depth, self.camera_spawnpoint, attach_to=self.ego_vehicle)

        # Spawn lidar and attach to vehicle
        bp_lidar = blueprint_library.find("sensor.lidar.ray_cast")
        bp_lidar.set_attribute("range", "120") # 120 meter range for cars and foliage
        bp_lidar.set_attribute("rotation_frequency", "10")
        bp_lidar.set_attribute("channels", "64") # vertical resolution of the laser scanner is 64
        bp_lidar.set_attribute("points_per_second", "1300000")
        bp_lidar.set_attribute("upper_fov", "2.0") # +2 up to -24.8 down
        bp_lidar.set_attribute("lower_fov", "-24.8")
        self.lidar_spawnpoint = carla.Transform(carla.Location(x=0, y=0, z=1.73))
        self.lidar = self.world.spawn_actor(bp_lidar, self.lidar_spawnpoint, attach_to=self.ego_vehicle)

        self.vehicle_manager.spawn_vehicles()

        self.tick_counter = 0

        self.RGB_colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
        self.BGR_colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0)]
        
        # Create opencv window
        cv2.namedWindow("inst_background", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("inst_background", 640, 360)

        if cfg.predict_lane:
            self.lanedet = LaneDet()
        else:
            self.lanemarkings = LaneMarkings(self.client, self.world)

        if cfg.predict_object:
            self.objectdet = ObjDet()


        wheelbase, rear_axle_offset = self.vehicle_manager.get_ego_vehicle_wheel()
        self.pure_pursuit = PurePursuit(self.camera_spawnpoint, wheelbase, rear_axle_offset, self.ego_vehicle)
        self.pid = PID()
        self.pid.update_setpoint(10) # m/s

        # Initialize Open3D Visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='CARLA LiDAR', width=800, height=600)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.random.rand(10, 3))
        self.vis.add_geometry(self.pcd)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
        self.vis.add_geometry(mesh_frame)

        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.asarray([0, 0, 0])
        render_opt.point_size = 1
        
        ctr = self.vis.get_view_control()
        ctr.change_field_of_view(step=90)
        ctr.set_constant_z_far(2000)
        ctr.set_constant_z_near(0.1)
        self.vis.reset_view_point(True)
        self.cam = ctr.convert_to_pinhole_camera_parameters()

        self.bbox_lines = []

        # Define line connections between the 8 corners
        self.lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom rectangle
            [4, 5], [5, 6], [6, 7], [7, 4],  # top rectangle
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]


    def reshape_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)) # BGRA
        array = array[:, :, :3] # BGR
        array = array[:, :, ::-1] # RGB
        return array # (H, W, C)


    def reshape_pointcloud(self, pointcloud):
        array = np.frombuffer(pointcloud.raw_data, dtype=np.float32)
        array = np.reshape(array, (-1, 4)).copy() # x, y, z, r
        return array # (N, 4) pointcloud
    

    def draw_image(self, surface, array, blend=False):
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1)) # (W, H, C)
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))


    def render_display(self, image_rgb, image_depth, lanes_list_processed, pointcloud, lidar_bboxes):
        # Draw the display.
        self.draw_image(self.display, image_rgb)
        
        inst_background = np.zeros_like(image_rgb)

        # Draw lane on pygame window and binary mask
        if(cfg.render_lanes):
            for i in range(len(lanes_list_processed)):
                for x, y, in lanes_list_processed[i]:
                    pygame.draw.circle(self.display, self.RGB_colors[i], (x, y), 3, 2)
                cv2.polylines(inst_background, np.int32([lanes_list_processed[i]]), isClosed=False, color=self.BGR_colors[i], thickness=5)                

        # Update point cloud
        self.pcd.points = o3d.utility.Vector3dVector(pointcloud)
        self.pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 1.0, 0.0], (pointcloud.shape[0], 1)))
        self.vis.update_geometry(self.pcd)

        # Clear previous bounding boxes
        for line in self.bbox_lines:
            self.vis.remove_geometry(line, reset_bounding_box=False)
        self.bbox_lines = []

        bboxes_corners = bbox3d2corners(lidar_bboxes)
        for corners in bboxes_corners:
            # Apply transformation to Open3D coordinate frame
            corners[:, 1] = -corners[:, 1] # convert from UE to Kitti/Open3D

            # Create LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(self.lines)

            # Set green color for all lines
            colors = [[0.0, 1.0, 0.0] for _ in range(len(self.lines))]
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # Add to visualizer and keep reference
            self.vis.add_geometry(line_set)
            self.bbox_lines.append(line_set)

        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        
        velocity = self.ego_vehicle.get_velocity() # m/s
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed = speed / 1000 * 3600 # convert to km/h
        text_surface = self.font.render(f'Speed: {speed:.1f} km/h', True, (0, 255, 0))
        self.display.blit(text_surface, (10, 10)) 
        
        pygame.display.flip()
        cv2.imshow("inst_background", inst_background)
        cv2.waitKey(1)


    def run(self):
        with CarlaSyncMode(self.world, self.tm, self.camera_rgb, self.camera_semseg, self.camera_depth, self.lidar, fps=cfg.fps) as sync_mode:
            try:
                while True:
                    ### pygame interaction ###
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            exit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_LEFT:
                                print("left")
                                self.tm.force_lane_change(self.ego_vehicle, False)
                            elif event.key == pygame.K_RIGHT:
                                print("right")
                                self.tm.force_lane_change(self.ego_vehicle, True)
                    

                    ### Manually run simulation ###
                    if not cfg.auto_run:
                        waiting = True
                        while waiting:
                            event = pygame.event.wait()
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                exit()
                            keys = pygame.key.get_pressed()
                            if keys[pygame.K_SPACE]:
                                waiting = False


                    ### Respawn stuck vehicle ###
                    if self.tick_counter % (cfg.respawn * cfg.fps) == 0:
                        self.vehicle_manager.check_vehicle()
                

                    ### Run simulation ###
                    self.pygame_clock.tick()
                    if cfg.predict_lane:
                        snapshot, sensor_rgb, sensor_depth, sensor_lidar = sync_mode.tick(timeout=1.0)
                    else:
                        snapshot, sensor_rgb, sensor_semseg, sensor_depth, sensor_lidar = sync_mode.tick(timeout=1.0)
                        sensor_semseg.convert(carla.ColorConverter.CityScapesPalette)
                        image_semseg = self.reshape_image(sensor_semseg)
                    
                    image_rgb = self.reshape_image(sensor_rgb)
                    image_depth = self.reshape_image(sensor_depth)
                    pointcloud = self.reshape_pointcloud(sensor_lidar)
                    self.tick_counter += 1


                    ### Get current waypoints ### 
                    waypoint = self.map.get_waypoint(self.ego_vehicle.get_location())
                    waypoint_list = []
                    for i in range(0, cfg.number_of_lanepoints):
                        waypoint_list.append(waypoint.next(i + cfg.meters_per_frame)[0])
                    if cfg.draw3DLanes:
                        for waypoint in waypoint_list:
                            self.world.debug.draw_point(location=waypoint.transform.location, size=0.05, life_time=2 * (1/cfg.fps), persistent_lines=False)                    
                    

                    ### Predict lanepoints for all lanes ###
                    if cfg.predict_lane:
                        img = Image.fromarray(image_rgb, mode="RGB") 
                        lanes_list_processed = self.lanedet.predict(img)
                    else:
                        lanes_list, x_lanes_list = self.lanemarkings.detect_lanemarkings(waypoint_list, image_semseg, self.camera_rgb)
                        lanes_list_processed = self.lanemarkings.lanemarkings_processed(lanes_list)
                    

                    ### Predict objects ###
                    if cfg.predict_object:
                        lidar_bboxes, labels = self.objectdet.predict(pointcloud)
                    else:
                        lidar_bboxes, labels = gt_bbox.get_bboxes(self.world, pointcloud, sensor_lidar)


                    if not cfg.carla_auto_pilot:
                        velocity = self.ego_vehicle.get_velocity() # m/s
                        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

                        ### Pure pursuit ###
                        steering_angle = self.pure_pursuit.run(lanes_list_processed, image_depth, speed)

                        ### PID ###
                        throttle = self.pid.compute_throttle(speed)

                        control = carla.VehicleControl(throttle=throttle, steer=steering_angle)
                        self.ego_vehicle.apply_control(control)

                        ### Obstacle Avoidance ###

                    ### Render display ###
                    pointcloud[:, 1] = -pointcloud[:, 1] # convert from UE to Kitti/Open3D
                    self.render_display(image_rgb, image_depth, lanes_list_processed, pointcloud[:, :3], lidar_bboxes)

            finally:
                self.vehicle_manager.destroy()
                for sensor in sync_mode.sensors:
                    if sensor:
                        sensor.destroy()
                print("Sensors destroyed")


if __name__ == '__main__':
    pygame.init()

    game = CarlaGame()
    game.run()

    pygame.quit()






