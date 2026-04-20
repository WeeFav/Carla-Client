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

        self.RGB_colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
        self.BGR_colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0)]
        
        # Create opencv window
        cv2.namedWindow("inst_background", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("inst_background", 640, 360)


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


    def render_display(self, image_rgb):
        # Draw the display.
        self.draw_image(self.display, image_rgb)
    
        pygame.display.flip()


    def run(self):
        with CarlaSyncMode(self.world, self.tm, self.camera_rgb, self.lidar, fps=cfg.fps) as sync_mode:
            try:
                while True:                    
                    ### Run simulation ###
                    self.pygame_clock.tick()
                    snapshot, sensor_rgb, sensor_lidar = sync_mode.tick(timeout=1.0)
                    
                    image_rgb = self.reshape_image(sensor_rgb)
                    pointcloud = self.reshape_pointcloud(sensor_lidar)

                    ### Render display ###
                    pointcloud[:, 1] = -pointcloud[:, 1] # convert from UE to Kitti/Open3D
                    self.render_display(image_rgb)

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