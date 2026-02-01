import torch
import numpy as np
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from torchvision.ops import nms
import sys
sys.path.append('/data/yyang/workspace/magiclidar/submodules')

class_names = ("car")

def colorstr(*input):
    """
        Helper function for style logging
    """
    *args, string = input if len(input) > 1 else ("bold", input[0])
    colors = {"bold": "\033[1m"}

    return "".join(colors[x] for x in args) + f"{string}"

class YOLO_WORLD:
    def __init__(self, config_file_path, pretrained_weight_path):
        cfg = Config.fromfile(
            config_file_path
        )
        cfg.work_dir = "/data/yyang/workspace/magiclidar/temp_test/workdir"
        cfg.load_from = pretrained_weight_path
        self.runner = Runner.from_cfg(cfg)
        self.runner.call_hook("before_run")
        self.runner.load_or_resume()
        self.pipeline = cfg.test_dataloader.dataset.pipeline
        self.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.runner.pipeline = Compose(self.pipeline)

        # run model evaluation
        self.runner.model.eval()

    def run_image(
            self,
            input_image = None,
            img_array = None,
            texts = None,
            max_num_boxes=100,
            score_thr=0.0,
            nms_thr=0.5,
            # output_image="output.png",
    ):
        # self.runner.model.reparameterize(texts)
        # texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]
        # output_image = "runs/detect/"+output_image
        if img_array is not None:
            data_info = self.runner.pipeline(dict(img_id=0, img=img_array,
                                            texts=texts))
        else:
            self.runner.pipeline = Compose(self.pipeline)
            data_info = self.runner.pipeline(dict(img_id=0, img_path=input_image,
                                            texts=texts))

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]
            self.runner.model.class_names = texts
            pred_instances = output.pred_instances

        # nms
        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]
        output.pred_instances = pred_instances

        # predictions
        pred_instances = pred_instances.cpu().numpy()
        if pred_instances.scores.shape[0] == 0:
            return np.zeros([4])
        top_1_idx = pred_instances.scores.argmax()

        return pred_instances.bboxes[top_1_idx]