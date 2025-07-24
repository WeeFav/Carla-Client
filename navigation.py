import carla
import random
import time
import sys
sys.path.append(r"C:\Users\marvi\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla")
# from agents.navigation.global_route_planner import GlobalRoutePlanner

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.load_world("Town03_Opt")

map = world.get_map()

# grp = GlobalRoutePlanner(map, 2)

# point_a = carla.Location(x=64.64, y=24.47, z=0)
# point_b = carla.Location(x=114.47, y=65.78, z=0)

# route = grp.trace_route(point_a, point_b)

# print(len(route))

# for waypoint in route:
#     world.debug.draw_string(location=waypoint[0].transform.location, text='^', life_time=120)

# waypoint_list = map.generate_waypoints(2.0)
# waypoint_tuple_list = map.get_topology()
# print(len(waypoint_list))
# print(len(waypoint_tuple_list))

# vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
# spawn_points = world.get_map().get_spawn_points()
# for i in range(0,50):
#     world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# for vehicle in world.get_actors().filter('*vehicle*'):
#     vehicle.set_autopilot(True)

time.sleep(100)

# for vehicle in world.get_actors().filter('*vehicle*'):
#     vehicle.destroy()