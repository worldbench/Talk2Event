import os
import argparse
import torch
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import utils.misc as utils
from datasets import build_dataset
from datasets.data_prefetcher import data_prefetcher

def parse_args():
    parser = argparse.ArgumentParser(description='Beauty DETR')
    parser.add_argument("--split", default='test')
    parser.add_argument("--attribute", default='relation_others')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    return parser.parse_args()

args = parse_args()
dataset_train = build_dataset('talk2event', image_set=args.split, args=args)
sampler_train = torch.utils.data.RandomSampler(dataset_train)
batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

data_loader_train = DataLoader(
    dataset_train,
    batch_sampler=batch_sampler_train,
    collate_fn=utils.collate_fn,
    num_workers=args.num_workers,
    pin_memory=True
)

device = torch.device(args.device)
prefetcher = data_prefetcher(data_loader_train, device, prefetch=True)
samples, event_samples, targets = prefetcher.next()

for i in range(len(data_loader_train)):
    print(f"Batch {i+1}:")
    samples, event_samples, targets = prefetcher.next()