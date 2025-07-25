from bev import BEV

class PurePursuit():
    def __init__(self, camera_spawnpoint, wheelbase, rear_axle_offset):
        self.bev = BEV(camera_spawnpoint)
        self.wheelbase = wheelbase
        self.rear_axle_offset = rear_axle_offset

    
    def get_centerline(self, left_lanemarking: list, right_lanemarking: list, lane_width):
        """
        lanemarkings should be in rear axle vehicle frame
        """
        if left_lanemarking and right_lanemarking:
            print(left_lanemarking)
            print(right_lanemarking)
    

    def run(self, lanes_list_processed, image_depth):
        lanes_rear = self.bev.pixel_to_rear(lanes_list_processed, image_depth, self.rear_axle_offset)
        centerline = self.get_centerline(lanes_rear[1].tolist(), lanes_rear[2].tolist(), None) # ego lanes