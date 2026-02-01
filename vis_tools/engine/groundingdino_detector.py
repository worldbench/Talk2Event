import sys
import torch.nn as nn
sys.path.append("/data/yyang/workspace/magiclidar")
from submodules.gdino.gdino import GroundingDino
from datasets import build_dataset

class GroundingDino_Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        self.init_detector()
    
    def init_detector(self):
        self.model = GroundingDino(
                config_path="/data/yyang/workspace/magiclidar/submodules/gdino/configs/GroundingDINO_SwinT_OGC.py",
                 pretrained_weight_path="submodules/pretrained/groundingdino_swint_ogc.pth", 
        )

    def build_dataset(self):
        dataset = build_dataset(self.config.combine_datasets_val[0], "test", self.config)
        self.dataset = dataset

    def forward(self, data_tuple):
        (samples, event_samples, targets) = data_tuple
        caption = targets[0]['caption']
        image = samples.tensors[0].permute(1,2,0).detach().cpu().numpy()
        pred_rel = self.model.run_image(image = image*255,
                                        text = caption)

        return pred_rel

if __name__ == "__main__":
    detector = GroundingDino_Detector()
    detector.infrence(0)
