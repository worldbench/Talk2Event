import torch
import argparse
import sys
import os
from tqdm import tqdm 
import numpy as np
from collections import defaultdict
import json
import pickle
sys.path.append("/data/dylu/project/butd_detr")
from datasets import build_dataset
import utils.misc as utils
from vis_tools.utils.model_dataset import get_args_parser
from vis_tools.engine import __all__ as all_detectors

CLASSES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle']

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes in [x, y, w, h] format."""
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1

    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

class T2E_Metric:
    '''
    Talk2Event Metric
    '''
    def __init__(self, threshold=[0.9, 0.95]):
        self.threshold = threshold
        self.class_total = defaultdict(int)
        self.class_iou_scores = defaultdict(list)
        self.num_objects_total = defaultdict(int)
        self.num_objects_iou_scores = defaultdict(list)

        for thr in threshold:
            setattr(self, f"class_acc{'{:02.0f}'.format(thr * 100)}", defaultdict(int))
            setattr(self, f"number_boxes_acc{'{:02.0f}'.format(thr * 100)}", defaultdict(int))

    def display_mertic(self):
        # miou
        miou_result = {}
        all_ious = []
        all_numbers = 0
        for class_name in CLASSES:
            iou_value = self.class_iou_scores[class_name]
            class_number = self.class_total[class_name]
            all_ious.extend(iou_value)
            all_numbers += class_number
            if class_number != 0:
                miou_result.update({f'{class_name}': "{:.2f}".format(100*sum(iou_value) / class_number)})
            else:
                miou_result.update({f'{class_name}': "{:.2f}".format(0)})
        if all_numbers != 0:
            miou_result.update({'mIoU': "{:.2f}".format(100*sum(all_ious) / all_numbers)})
        else:
            miou_result.update({'mIoU': "{:.2f}".format(0)})

        # acc
        acc_result = {}
        for thr in self.threshold:
            thr_all_numbers = 0
            thr_miou_result = {}
            for class_name in CLASSES:
                class_number = getattr(self, f"class_acc{'{:02.0f}'.format(thr * 100)}")[class_name]
                thr_all_numbers += class_number
                if class_number != 0:
                    thr_miou_result.update({f'{class_name}': "{:.2f}".format(100*class_number / self.class_total[class_name])})
                else:
                    thr_miou_result.update({f'{class_name}': "{:.2f}".format(0)})

            if all_numbers != 0:
                thr_miou_result.update({'ALL': "{:.2f}".format(100*thr_all_numbers / all_numbers)})
            else:
                thr_miou_result.update({'ALL': "{:.2f}".format(0)})

            acc_result.update({f"thr_acc{'{:02.0f}'.format(thr * 100)}": thr_miou_result})

        print("======================IoU======================")
        for key ,value in miou_result.items():
            print(f"{key}:{value}")
        print("======================Acc======================")
        for key ,value in acc_result.items():
            print(f"{key}:{value}")
        return miou_result, acc_result

    def record_single(self, rel_dict):
        pred_box = rel_dict['pred_box']
        gt_box = rel_dict['gt_box']
        gt_class = rel_dict['gt_class']
        num_objects = rel_dict['other_num_objects']
        iou = compute_iou(pred_box, gt_box)

        self.class_total[gt_class] += 1
        self.num_objects_total[num_objects] += 1
        self.class_iou_scores[gt_class].append(iou)
        self.num_objects_iou_scores[num_objects].append(iou)

        for thr in self.threshold:
            if iou >= thr:
                recoder = getattr(self, f"class_acc{'{:02.0f}'.format(thr * 100)}")
                recoder[gt_class] += 1
                recoder = getattr(self, f"number_boxes_acc{'{:02.0f}'.format(thr * 100)}")
                recoder[num_objects] += 1

class Tester:
    def __init__(self, detector_name, batch_size=16, num_object_list=[1,2,3,4,5,6,7,8,9,10]):
        self.detector_name = detector_name
        self.batch_size = batch_size
        self.num_object_list = num_object_list
        self.device = 'cuda'
        self.metric_recoder = T2E_Metric()
        for object_num in num_object_list:
            setattr(self, f'metric_recoder_{str(object_num).zfill(2)}', T2E_Metric())
        self.init_args()
        self.build_model()
        self.build_dataloader()
        self.model.eval()

    def init_args(self):
        parser = argparse.ArgumentParser('Deformable', parents=[get_args_parser()], allow_abbrev=False )
        # args = parser.parse_args()
        args, unknown = parser.parse_known_args()

        args.output_dir = "exps/status_event"
        args.dataset_config = "configs/pretrain.json"
        args.batch_size = 2
        args.lr = 1e-5
        args.lr_backbone = 1e-6
        args.text_encoder_lr = 6e-6
        args.weight_decay = 1e-4
        args.large_scale = True
        args.save_freq = 1
        args.eval_skip = 1
        args.ema
        args.combine_datasets_val = ["talk2event"]
        args.resume = "exps/relation_others_image/checkpoint0019.pth"
        args.eval
        args.attribute = 'appearance'
        args.event_config = 'models/event/backbone.yaml'
        args.event_checkpoint = 'data/flexevent.ckpt'
        args.modality = 'image'

        self.config = args

    def build_model(self):
        self.model = all_detectors[self.detector_name]()

    def build_dataloader(self):
        self.dataset = build_dataset(self.config.combine_datasets_val[0], "test", self.config)
        
    def post_process(self, outputs, targets, image_size):
        rel_list = []
        gt_bboxes = torch.cat([item['boxes'] for item in targets], dim=0)
        rel_dict=dict({
            'pred_box': outputs,
            'gt_box': gt_bboxes[0].detach().cpu().numpy(),
            'gt_class': targets[0]['category'],
            'other_num_objects': targets[0]['other_num_objects'] + 1,
            'image_path': targets[0]['image_path'],
            'event_path': targets[0]['event_path'],
            'caption': targets[0]['caption']
        })
        rel_list.append(rel_dict)
        return rel_list

    def test(self):
        num_steps = int(len(self.dataset))
        # num_steps = 100
        output_list = []
        for i in tqdm(range(num_steps)):
            (samples, event_samples, targets) = utils.collate_fn([self.dataset.__getitem__(i, custom_aug=True)])
            # if self.detector_name == 'yolo_world':
            # try:
            outputs = self.model((samples, event_samples, targets))
            # except:
            #     outputs = np.zeros([4])
            #     print(f'Long token of sample: {i}')

            sub_output_list = self.post_process(outputs, targets, image_size=samples.tensors.shape[2:])
            output_list.extend(sub_output_list)

        # save inference results as pkl
        with open(f"exps/record_results/{self.detector_name}_image_{self.config.attribute}_results.pkl", 'wb') as f:
            pickle.dump(output_list, f)

        self.calculate_metric(output_list)

    def calculate_metric(self, output_list):
        for idx in range(len(output_list)):
            output = output_list[idx]
            self.metric_recoder.record_single(output)
            object_num = output['other_num_objects']
            try:
                record = getattr(self, f'metric_recoder_{str(object_num).zfill(2)}')
                record.record_single(output)
            except:
                record = getattr(self, f'metric_recoder_{str(10).zfill(2)}')
                record.record_single(output)
        final_rel_dict = {}
        miou_result, acc_result = self.metric_recoder.display_mertic()
        final_rel_dict.update({
            'overall_metrics': {
                'iou_results': miou_result,
                'acc_results': acc_result
            }
        })
        for object_num in self.num_object_list:
            print(f'************************ #{object_num}# ************************')
            record = getattr(self, f'metric_recoder_{str(object_num).zfill(2)}')
            miou_result, acc_result = record.display_mertic()
            final_rel_dict.update({
                f'{object_num}_metrics': {
                    'iou_results': miou_result,
                    'acc_results': acc_result
                }
            })
        with open(f"exps/record_results/{self.detector_name}_image_{self.config.attribute}_results.json", 'w', encoding='utf-8') as f:
            json.dump(final_rel_dict, f, ensure_ascii=False, indent=4)


    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tester = Tester(detector_name='owl_vitv2', batch_size=1)
    tester.test()