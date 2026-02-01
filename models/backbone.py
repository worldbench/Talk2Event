# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2021 Carnegie Mellon University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import numpy as np

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
import ipdb
st  = ipdb.set_trace

import yaml
from dotmap import DotMap
from .event.maxvit_rnn import RNNDetector as MaxViTRNNDetector
from .event.utils import _get_modified_hw_multiple_of

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool,
                    butd=False, with_learned_class_embeddings=False,
                    embeddings_path=None, device=None):
        super().__init__()
        self.butd = butd
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        if butd:
            if with_learned_class_embeddings:
                self.butd_class_embeddings = nn.Embedding(1601, 768)

                saved_class_embeddings = torch.from_numpy(
                                np.load(embeddings_path, allow_pickle=True))
                saved_class_embeddings = torch.cat([
                    torch.zeros(1, 768),
                    saved_class_embeddings
                ], dim=0)
                self.butd_class_embeddings.weight.data.copy_(saved_class_embeddings)
                self.butd_class_embeddings.requires_grad = False
            else:
                num_detector_classes = 1601
                self.butd_class_embeddings = nn.Embedding(num_detector_classes, 32)

    def forward(self, tensor_list: NestedTensor):
        # Dict: {'0', '1', '2'}
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
     
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class EventBackbone(nn.Module):

    def __init__(self, args):
        super().__init__()

        with open(args.event_config, "r") as f:
            config = yaml.safe_load(f)

        config = DotMap(config)
        backbone_cfg = config.backbone

        partition_split_32 = backbone_cfg.partition_split_32
        assert partition_split_32 in (1, 2, 4)

        multiple_of = 32 * partition_split_32
        mdl_hw = _get_modified_hw_multiple_of(hw=(480,640), multiple_of=multiple_of)
        backbone_cfg.in_res_hw = mdl_hw

        attention_cfg = backbone_cfg.stage.attention
        partition_size = tuple(x // (32 * partition_split_32) for x in mdl_hw)
        assert (mdl_hw[0] // 32) % partition_size[0] == 0, f'{mdl_hw[0]=}, {partition_size[0]=}'
        assert (mdl_hw[1] // 32) % partition_size[1] == 0, f'{mdl_hw[1]=}, {partition_size[1]=}'
        print(f'Set partition sizes: {partition_size}')
        attention_cfg.partition_size = partition_size
        
        self.event_backbone = MaxViTRNNDetector(backbone_cfg)

        #load checkpoint to args.device

        checkpoint = torch.load(args.event_checkpoint, map_location=args.device)
        event_ckpt = checkpoint["state_dict"]
        prefix = "mdl.backbone.stages"
        filtered_ckpt = {
                    k[len('mdl.backbone')+1:]: v        # +1 to drop the trailing dot
                    for k, v in event_ckpt.items()
                    if k.startswith(prefix + ".")
        }
        missing, unexpected = self.event_backbone.load_state_dict(filtered_ckpt, strict = False)
        print(" =======event backbone missing keys======== : ", missing)
        print(" =======event backbone unexpected keys======: ", unexpected)
        self.strides = [4, 2, 2, 2]
        self.num_channels = [64, 128, 256, 512]

    def forward(self, tensor_list: NestedTensor):
        # Dict: {'0', '1', '2'}
        xs = self.event_backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
     
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 butd = False,
                 with_learned_class_embeddings=False,
                 embeddings_path=None, device=None):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers,
                                butd=butd,
                                with_learned_class_embeddings=with_learned_class_embeddings,
                                embeddings_path=embeddings_path, device=device)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.backbone = backbone
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels # [512, 1024, 2048]

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone,
             return_interm_layers, args.dilation, args.butd,
             args.with_learned_class_embeddings,
             args.embeddings_path, args.device)
    model = Joiner(backbone, position_embedding)
    return model

def build_event_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = EventBackbone(args)
    model = Joiner(backbone, position_embedding)
    return model