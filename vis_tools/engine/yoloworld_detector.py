import argparse
import sys
import torch.nn as nn
sys.path.append('/data/yyang/workspace/magiclidar/submodules')
sys.path.append("/data/yyang/workspace/magiclidar")
from yolo_world.yolo_world import YOLO_WORLD
from datasets import build_dataset

class YoloWorld_Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        self.init_detector()
    
    def init_detector(self):
        self.model = YOLO_WORLD(
            config_file_path="/data/yyang/workspace/magiclidar/submodules/yolo_world/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py",
            pretrained_weight_path="/data/yyang/workspace/magiclidar/submodules/pretrained/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
        )

    def build_dataset(self):
        dataset = build_dataset(self.config.combine_datasets_val[0], "test", self.config)
        self.dataset = dataset

    def forward(self, data_tuple):
        (samples, event_samples, targets) = data_tuple
        caption = targets[0]['caption']
        image = samples.tensors[0].permute(1,2,0).detach().cpu().numpy()
        pred_rel = self.model.run_image(img_array = image*255,
                                        texts = [[caption] + [' ']])
                                        # texts = [['car'] + ['']])

        return pred_rel

if __name__ == "__main__":
    detector = YoloWorld_Detector()
    detector.infrence(0)
