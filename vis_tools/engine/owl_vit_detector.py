import sys
import torch.nn as nn
sys.path.append("/data/yyang/workspace/magiclidar")
from submodules.owl_vit.owl_vitv2 import OWL_VOTV2
from datasets import build_dataset

class OWL_VIT_Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        self.init_detector()
    
    def init_detector(self):
        self.model = OWL_VOTV2()

    def build_dataset(self):
        dataset = build_dataset(self.config.combine_datasets_val[0], "test", self.config)
        self.dataset = dataset

    def forward(self, data_tuple):
        (samples, event_samples, targets) = data_tuple
        caption = targets[0]['attributes']['appearance'][0]
        # caption = targets[0]['caption']
        image = samples.tensors[0].permute(1,2,0).detach().cpu().numpy()
        pred_rel = self.model.run_image(image = image*255,
                                        text = caption)

        return pred_rel

class OWL_VIT_V2_Detector(OWL_VIT_Detector):
    def __init__(self):
        super().__init__()

    def init_detector(self):
        self.model = OWL_VOTV2()

if __name__ == "__main__":
    detector = OWL_VIT_Detector()
    detector.infrence(0)
