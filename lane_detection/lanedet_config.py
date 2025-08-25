# Camera
img_w = 1280
img_h = 720

# Network
griding_num = 100
cls_num_per_lane = 56 # number of row anchors
num_lanes = 4
num_cls = 4
backbone = '18'
model_path = "./lane_detection/ep049.pth"
use_classification = True

carla_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]