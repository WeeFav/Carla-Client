import glob
import os
import sys
import carla
import random
import pygame
import numpy as np
import open3d as o3d
from carla_sync_mode import CarlaSyncMode
import threading
import queue

lidar_queue = queue.Queue(maxsize=1)

def open3d_thread():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='CARLA LiDAR', width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(10, 3))
    vis.add_geometry(pcd)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0, 0, 0])
    render_opt.point_size = 1

    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=90)
    ctr.set_constant_z_far(2000)
    ctr.set_constant_z_near(0.1)
    vis.reset_view_point(True)

    while True:
        # Non-blocking check for new point cloud data
        try:
            pointcloud = lidar_queue.get_nowait()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile([1.0, 1.0, 0.0], (pointcloud.shape[0], 1))
            )
            vis.update_geometry(pcd)
        except queue.Empty:
            pass

        if not vis.poll_events():
            break
        vis.update_renderer()

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def reshape_pointcloud(pointcloud):
    array = np.frombuffer(pointcloud.raw_data, dtype=np.float32)
    array = np.reshape(array, (-1, 4)).copy() # x, y, z, r
    return array # (N, 4) pointcloud

def main():
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()
    tm = client.get_trafficmanager()

    o3d_thread = threading.Thread(target=open3d_thread, daemon=True)
    o3d_thread.start()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        # Spawn lidar and attach to vehicle
        bp_lidar = blueprint_library.find("sensor.lidar.ray_cast")
        bp_lidar.set_attribute("range", "120") # 120 meter range for cars and foliage
        bp_lidar.set_attribute("rotation_frequency", "20")
        bp_lidar.set_attribute("channels", "64") # vertical resolution of the laser scanner is 64
        bp_lidar.set_attribute("points_per_second", "1300000")
        bp_lidar.set_attribute("upper_fov", "2.0") # +2 up to -24.8 down
        bp_lidar.set_attribute("lower_fov", "-24.8")
        lidar_spawnpoint = carla.Transform(carla.Location(x=0, y=0, z=1.73))
        lidar = world.spawn_actor(bp_lidar, lidar_spawnpoint, attach_to=vehicle)
        actor_list.append(lidar)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, tm, camera_rgb, camera_semseg, lidar, fps=20) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                # 1. Advance simulation
                snapshot, image_rgb, image_semseg, sensor_lidar = sync_mode.tick(timeout=2.0)

                # 2. Process Point Cloud for Open3D
                pointcloud = reshape_pointcloud(sensor_lidar)
                pointcloud[:, 1] = -pointcloud[:, 1] # Flip Y for Open3D coordinates
                
                try:
                    lidar_queue.put_nowait(pointcloud)
                except queue.Full:
                    pass  # Skip frame, Open3D is still processing previous one

                # 3. PRE-PROCESS Images (Crucial: Convert before drawing)
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                
                # 4. Update Vehicle (Optional: move this before or after rendering)
                waypoint = random.choice(waypoint.next(1.5))
                vehicle.set_transform(waypoint.transform)

                # 5. Draw to Pygame
                # Clear the screen with a visible color (like blue) to test if Pygame is rendering AT ALL
                display.fill((0, 0, 50)) 

                draw_image(display, image_rgb)
                draw_image(display, image_semseg, blend=True)

                # Calculate FPS
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))

                # Force Pygame to update the window
                pygame.display.flip()
    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
