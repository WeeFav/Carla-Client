import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import scipy.special
import numpy as np
from PIL import Image

from .model import UFLDNet
from . import lanedet_config as cfg

class LaneDet():
    def __init__(self):
        self.model = UFLDNet(
            pretrained=False,
            backbone=cfg.backbone,
            cls_dim=(cfg.griding_num + 1, cfg.cls_num_per_lane, cfg.num_lanes),
            cat_dim=(cfg.num_lanes, cfg.num_cls),
            use_aux=False, # we dont need auxiliary segmentation in testing
            use_classification=cfg.use_classification
        )
        self.model.cuda()

        # load model weights
        state_dict = torch.load(cfg.model_path, map_location='cuda')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v
                
        self.model.load_state_dict(compatible_state_dict, strict=False)
        self.model.eval()

        # transform input image
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    

    def predict(self, img: Image):
        # img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # convert (H, W, C) to [C, H, W], range [0,1]
        # img_tensor = TF.resize(img_tensor, (288, 800))  # resize using functional API
        # img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = self.img_transforms(img)
        img = img.unsqueeze(dim=0).cuda()

        with torch.no_grad():
            out = self.model(img) # (batch_size, num_gridding, num_cls_per_lane, num_of_lanes)

        detection = out['det']
        
        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = detection[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :] # flips rows
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) # removes the last class, which is often reserved for no lane / background.
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0) # expectation / avg idx
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc # (56, 4)

        lanes_list = [[] for _ in range(4)]
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        p = (int(out_j[k, i] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.carla_row_anchor[cfg.cls_num_per_lane - 1 - k] / 288)) - 1)
                        lanes_list[i].append(p)
                

        return lanes_list





        
