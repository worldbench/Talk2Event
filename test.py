import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import sys
import os
from tqdm import tqdm 
from collections import defaultdict
import json
import pickle
sys.path.append("/data/dylu/project/butd_detr")
from models import build_bdetr_model
from datasets import build_dataset
import utils.misc as utils
from datasets.data_prefetcher import targets_to
from vis_tools.utils.common import rescale_bboxes
from vis_tools.utils.model_dataset import get_args_parser
from datasets.data_prefetcher import data_prefetcher

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
        miou_result.update({'mIoU': "{:.2f}".format(100*sum(all_ious) / all_numbers)})
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

            thr_miou_result.update({'ALL': "{:.2f}".format(100*thr_all_numbers / all_numbers)})
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
    def __init__(self, batch_size=16, num_object_list=[1,2,3,4,5,6,7,8,9,10]):
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

        args.output_dir = "/dataset/yyang/magiclidar/log/all_fusion"
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
        args.resume = "/dataset/yyang/magiclidar/log/all_fusion/checkpoint0015.pth"
        args.eval
        args.attribute = 'all'
        args.event_config = 'models/event/backbone.yaml'
        args.event_checkpoint = 'data/flexevent.ckpt'
        args.modality = 'fusion'
        args.moe_fusion = False

        self.config = args

    def build_model(self):
        model, _, _ = \
            build_bdetr_model(self.config)
        model.to(self.device)

        checkpoint = torch.load(self.config.resume, map_location='cpu')
        model.load_state_dict(checkpoint["model_ema"], strict=False)
        model.eval()
        self.model = model

    @property
    def split_checkpoint(self):
        checkpoints = self.config.resume.split('/')[-1]
        return checkpoints.split('.')[0]

    def build_dataloader(self):
        dataset = build_dataset(self.config.combine_datasets_val[0], "test", self.config)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=utils.collate_fn,
            drop_last=False,
            pin_memory=True,
        )

    def post_process(self, outputs, targets, image_size):
        rel_list = []
        positive_map_list = [item['positive_map'] for item in targets]
        posneg_map = torch.cat(positive_map_list, dim=0) # B X N
        if len(outputs['pred_logits']) == 4:
            max_gate = outputs["max_gate"]
            print(max_gate.float().mean())

            batch_idx = torch.arange(max_gate.size(0), device=max_gate.device)  # [0,1,...,B-1]
            logits = outputs['pred_logits'][max_gate, batch_idx] 
            positive_map_tensor = torch.stack(positive_map_list)  
            posneg_map = positive_map_tensor[batch_idx, max_gate]
            # logits, _ = outputs['pred_logits'].max(dim=0)
        else:
            logits = outputs['pred_logits'] # B 256

        if not self.config.moe_fusion:
            prob = F.softmax(logits, -1)
            scores_ = torch.bmm(prob, posneg_map.unsqueeze(1).permute(0, 2, 1)) # B X Q X N
            scores, labels = scores_.max(-1)

            # probas = 1 - logits.softmax(-1)[:, :, -1].cpu()
            keep = scores.argmax(dim=-1)
            expanded_idx = keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4)
            pred_boxes = torch.gather(outputs['pred_boxes'].cpu(), 1, expanded_idx.cpu()).squeeze(1)

        else:
            pmap = torch.stack(positive_map_list, dim=0) # B obj 256
            scores = (logits.unsqueeze(1) * pmap.unsqueeze(2)).sum(-1) # (B, obj, Q)
            top = scores.argsort(2, True)[:, :, 0]
            indices_expanded = top.unsqueeze(-1).expand(-1, -1, outputs['pred_boxes'].size(2))  # [B, M, C]
            pred_boxes = torch.gather(outputs['pred_boxes'], dim=1, index=indices_expanded)
            pred_boxes = pred_boxes[:,0,:].cpu()
            
        gt_bboxes = torch.cat([item['boxes'] for item in targets], dim=0)

        bboxes_scaled = rescale_bboxes(pred_boxes, image_size).detach().cpu().numpy()
        gt_bboxes_scaled = rescale_bboxes(gt_bboxes.cpu(), image_size).detach().cpu().numpy()
        for batch_id in range(bboxes_scaled.shape[0]):
            rel_dict=dict({
                'pred_box': bboxes_scaled[batch_id],
                'gt_box': gt_bboxes_scaled[batch_id],
                'gt_class': targets[batch_id]['category'],
                'other_num_objects': targets[batch_id]['other_num_objects'] + 1,
                'image_path': targets[batch_id]['image_path'],
                'event_path': targets[batch_id]['event_path'],
                'caption': targets[batch_id]['caption']
            })
            rel_list.append(rel_dict)
        return rel_list

    def test(self):
        prefetcher = data_prefetcher(self.dataloader, self.device, prefetch=True)
        num_steps = int(len(self.dataloader))
        # num_steps = 100
        output_list = []
        for i in tqdm(range(num_steps)):
            samples, event_samples, targets = prefetcher.next()
            samples = samples.to(self.device)
            event_samples = event_samples.to(self.device)
            targets = targets_to(targets, self.device)
            captions = [t["caption"] for t in targets]
            positive_map = torch.cat(
                [t["positive_map"] for t in targets])

            memory_cache = None
            butd_boxes = None
            butd_masks = None
            butd_classes = None
            if self.config.butd:
                butd_boxes = torch.stack([t['butd_boxes'] for t in targets], dim=0)
                butd_masks = torch.stack([t['butd_masks'] for t in targets], dim=0)
                butd_classes = torch.stack([t['butd_classes'] for t in targets], dim=0)
            with torch.no_grad():
                memory_cache = self.model(
                    samples,
                    event_samples,
                    captions,
                    encode_and_save=True,
                    butd_boxes=butd_boxes,
                    butd_classes=butd_classes,
                    butd_masks=butd_masks
                )
                outputs = self.model(
                    samples, event_samples, captions, encode_and_save=False,
                    memory_cache=memory_cache,
                    butd_boxes=butd_boxes,
                    butd_classes=butd_classes,
                    butd_masks=butd_masks,
                    targets = targets
                )
            sub_output_list = self.post_process(outputs, targets, image_size=samples.tensors.shape[2:])
            output_list.extend(sub_output_list)

        # save inference results as pkl
        with open(f"{self.config.output_dir}/{self.split_checkpoint}.pkl", 'wb') as f:
        # with open(f"exps/record_results/butd_{self.config.modality}_results.pkl", 'wb') as f:
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
        with open(f"{self.config.output_dir}/{self.split_checkpoint}.json", 'w', encoding='utf-8') as f:
        # with open(f"exps/record_results/butd_{self.config.modality}_{self.split_checkpoint}.json", 'w', encoding='utf-8') as f:
            json.dump(final_rel_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    tester = Tester(batch_size=16)
    tester.test()
    # 1. ######################### number objects
    # 1. ######################### number objects

    # 100 个最差的
    # 
    # ssh -X -L 7007:localhost:7007 -L 8020:localhost:8020 yyang@cvrp-gpu-6.d2.comp.nus.edu.sg
