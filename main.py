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

import config as cfg
from carla_sync_mode import CarlaSyncMode
from vehicle_manager import VehicleManager
from lane_detection.lanemarkings import LaneMarkings
from lane_detection.lanedet import LaneDet
from object_detection.objectdet import ObjDet
from object_detection import gt_bbox
from controllers.pure_pursuit import PurePursuit

class CarlaGame():
    def __init__(self):
        self.display = pygame.display.set_mode((cfg.image_width, cfg.image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()

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


    def reshape_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)) # BGRA
        array = array[:, :, :3] # BGR
        array = array[:, :, ::-1] # RGB
        return array # (H, W, C)


    def reshape_pointcloud(self, pointcloud):
        array = np.frombuffer(pointcloud.raw_data, dtype=np.float32)
        array = np.reshape(array, (-1, 4)) # x, y, z, r
        return array # (N, 4) pointcloud
    

    def draw_image(self, surface, array, blend=False):
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1)) # (W, H, C)
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))


    def render_display(self, image_rgb, image_depth, lanes_list_processed):
        # Draw the display.
        self.draw_image(self.display, image_rgb)
        
        inst_background = np.zeros_like(image_rgb)

        # Draw lane on pygame window and binary mask
        if(cfg.render_lanes):
            for i in range(len(lanes_list_processed)):
                for x, y, in lanes_list_processed[i]:
                    pygame.draw.circle(self.display, self.RGB_colors[i], (x, y), 3, 2)
                cv2.polylines(inst_background, np.int32([lanes_list_processed[i]]), isClosed=False, color=self.BGR_colors[i], thickness=5)                

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
                        lidar_bboxes, labels = gt_bbox.get_bboxes(pointcloud, sensor_lidar)


                    if not cfg.carla_auto_pilot:
                        ### Pure pursuit ###
                        self.pure_pursuit.run(lanes_list_processed, image_depth)

                        ### Obstacle Avoidance ###


                    ### Render display ###
                    self.render_display(image_rgb, image_depth, lanes_list_processed)

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






