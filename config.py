import carla

# Camera
fps = 10
image_width = 1280
image_height = 720
fov = 90

# Lanes
meters_per_frame = 1.0
number_of_lanepoints = 80
junctionMode = True
render_lanes = True
draw3DLanes = False

row_anchor_start = 160
h_samples = []
for y in range(row_anchor_start, image_height, 10):
	h_samples.append(y)

# World
town = 'Town10HD_Opt'
num_vehicles = 50
# weather = carla.WeatherParameters(
# 	cloudiness=0,
# 	precipitation=0,
# 	precipitation_deposits=0,
# 	sun_altitude_angle=-90,
# 	fog_density=0,
# 	fog_distance=0,
# 	wetness=0
# )
weather = carla.WeatherParameters.ClearNoon

# Mode
auto_run = True
respawn = 50 # in seconds
carla_auto_pilot = True
predict_lane = False
predict_object = True