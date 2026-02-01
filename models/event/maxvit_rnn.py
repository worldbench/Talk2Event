from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
# from detectron2.modeling import build_backbone
# from detectron2.structures import ImageList
# from detectron2.structures import Boxes
# from detectron2.modeling.poolers import ROIPooler
from torch.nn import TransformerDecoder, TransformerDecoderLayer

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

# from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates
from .layers.rnn import DWSConvLSTM2d
from .layers.maxvit.maxvit import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    get_downsample_layer_Cf2Cl,
    PartitionType)

# from models.layers.maxvit.moe import MoEConv
from .base import BaseDetector


class RNNDetector(BaseDetector):
    def __init__(self, mdl_config):
        super().__init__()

        ###### Config ######
        in_channels = mdl_config.input_channels#20
        embed_dim = mdl_config.embed_dim#64
        dim_multiplier_per_stage = tuple(mdl_config.dim_multiplier)
        num_blocks_per_stage = tuple(mdl_config.num_blocks)#[1, 1, 1, 1]
        T_max_chrono_init_per_stage = tuple(mdl_config.T_max_chrono_init)#[4, 8, 16, 32]
        enable_masking = mdl_config.enable_masking

        num_stages = len(num_blocks_per_stage)
        assert num_stages == 4

        assert isinstance(embed_dim, int)
        assert num_stages == len(dim_multiplier_per_stage)
        assert num_stages == len(num_blocks_per_stage)
        assert num_stages == len(T_max_chrono_init_per_stage)

        ###### Compile if requested ######
        compile_cfg = mdl_config.compile
        if compile_cfg is not None:
            compile_mdl = compile_cfg.enable
            if compile_mdl and th_compile is not None:
                compile_args = OmegaConf.to_container(compile_cfg.args, resolve=True, throw_on_missing=True)
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print('Could not compile backbone because torch.compile is not available')
        ##################################

        #image backbone
        # self.image_backbone = build_backbone(mdl_config.image_backbone)
        # for param in self.image_backbone.parameters():
        #     param.requires_grad = False
        # self.in_features = mdl_config.image_backbone.MODEL.ROI_HEADS.IN_FEATURES
        # self.size_divisibility = self.image_backbone.size_divisibility

        # pixel_mean = th.Tensor(mdl_config.image_backbone.MODEL.PIXEL_MEAN).view(3, 1, 1)
        # pixel_std = th.Tensor(mdl_config.image_backbone.MODEL.PIXEL_STD).view(3, 1, 1)
        # self.normalizer = lambda x: (x - pixel_mean.to(x.device)) / pixel_std.to(x.device)

        # #bbox padding
        # self.bbox_padding = 35

        # #bbox pooler
        # in_features = mdl_config.image_backbone.MODEL.ROI_HEADS.IN_FEATURES
        # pooler_scales = tuple(1.0 / self.image_backbone.output_shape()[k].stride for k in in_features)

        # self.box_pooler = ROIPooler(
        #     output_size=mdl_config.image_backbone.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
        #     scales=pooler_scales,
        #     sampling_ratio=mdl_config.image_backbone.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
        #     pooler_type=mdl_config.image_backbone.MODEL.ROI_BOX_HEAD.POOLER_TYPE,
        # )

        input_dim = in_channels
        patch_size = mdl_config.stem.patch_size#4
        stride = 1
        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]#64,128,256,512
        
        image_fusion = [True, True, True, True]
        self.stages = nn.ModuleList()
        self.strides = []
        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
                enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            spatial_downsample_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]
            enable_masking_in_stage = enable_masking and stage_idx == 0
            stage = RNNDetectorStage(dim_in=input_dim,
                                     stage_dim=stage_dim,
                                     spatial_downsample_factor=spatial_downsample_factor,
                                     num_blocks=num_blocks,
                                     enable_token_masking=enable_masking_in_stage,
                                     T_max_chrono_init=T_max_chrono_init_stage,
                                     stage_cfg=mdl_config.stage
                                     )
            stride = stride * spatial_downsample_factor
            self.strides.append(stride)

            input_dim = stage_dim
            self.stages.append(stage)

        self.num_stages = num_stages

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.stage_dims[stage_idx] for stage_idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.strides[stage_idx] for stage_idx in stage_indices)

    # def preprocess_image(self, batched_inputs):
    #     """
    #     Normalize, pad and batch the input images.
    #     """
    #     images = [self.normalizer(x) for x in batched_inputs]
    #     images = ImageList.from_tensors(images, self.size_divisibility)# image tensor 2,3,224,320

    #     images_whwh = list()
    #     for bi in batched_inputs:
    #         h, w = bi.shape[-2:]
    #         images_whwh.append(th.tensor([w, h, w, h], dtype=th.float32, device=images.device))
    #     images_whwh = th.stack(images_whwh)

    #     return images, images_whwh
    
    def forward(self, x, prev_states = None, token_mask = None):

        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages

        states = list()
        output = {}
        # loss_output = {}

        # device = x.device
        # # process image
        # images, images_whwh = self.preprocess_image(image)

        # # Feature Extraction. features[2x256x56x80, 2x256x28x40, 2x256x14x20, 2x256x7x10]
        # src = self.image_backbone(images.tensor)
        # features = list()
        # for f in self.in_features:
        #     feature = src[f]
        #     features.append(feature)

        # weights = []
        for stage_idx, stage in enumerate(self.stages):
            # x, state = stage(x, prev_states[stage_idx], token_mask if stage_idx == 0 else None, roi_features, roi_mask)
            x = stage(x, prev_states[stage_idx], token_mask if stage_idx == 0 else None)
            # states.append(state)
            stage_number = stage_idx + 1
            output[stage_number] = x
            # stage_number = stage_idx + 1
            # output[stage_number] = moe_x
            # loss_output[stage_number] = moe_loss
        #     weights.append(weight)
        # weights = th.stack(weights)
        # weights = weights.mean(0)
        return output


class MaxVitAttentionPairCl(nn.Module):
    def __init__(self,
                 dim,
                 skip_first_norm,
                 attention_cfg):
        super().__init__()

        self.att_window = PartitionAttentionCl(dim=dim,
                                               partition_type=PartitionType.WINDOW,
                                               attention_cfg=attention_cfg,
                                               skip_first_norm=skip_first_norm)
        self.att_grid = PartitionAttentionCl(dim=dim,
                                             partition_type=PartitionType.GRID,
                                             attention_cfg=attention_cfg,
                                             skip_first_norm=False)

    def forward(self, x):
        x = self.att_window(x)
        x = self.att_grid(x)
        return x


class RNNDetectorStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self,
                 dim_in,
                 stage_dim,
                 spatial_downsample_factor,
                 num_blocks,
                 enable_token_masking,
                 T_max_chrono_init,
                 stage_cfg
                 ):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        downsample_cfg = stage_cfg.downsample
        lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention
        # self.image_fusion = image_fusion
        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in,
                                                           dim_out=stage_dim,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg)
        blocks = [MaxVitAttentionPairCl(dim=stage_dim,
                                        skip_first_norm=i == 0 and self.downsample_cf2cl.output_is_normed(),
                                        attention_cfg=attention_cfg) for i in range(num_blocks)]
        self.att_blocks = nn.ModuleList(blocks)
        self.lstm = DWSConvLSTM2d(dim=stage_dim,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))

        ###### Mask Token ################
        self.mask_token = nn.Parameter(th.zeros(1, 1, 1, stage_dim),
                                       requires_grad=True) if enable_token_masking else None
        if self.mask_token is not None:
            th.nn.init.normal_(self.mask_token, std=.02)
        # ##################################
        # if self.image_fusion:
        #     self.moe_conv_layer = MoEConv(M=2, d=2*stage_dim, K=2)
        #     self.embedder = nn.Conv2d(in_channels=256, out_channels=stage_dim, kernel_size=1, stride=1, padding=0)
        #     self.embedder = Mlp(in_features=256, hidden_features=stage_dim, out_features=stage_dim)
        #     decoder_layer = TransformerDecoderLayer(d_model=stage_dim, nhead=4, dim_feedforward=stage_dim)
        #     self.decoder = TransformerDecoder(decoder_layer, 4)

    def forward(self, x,
                h_and_c_previous = None,
                token_mask = None):
        
        x = self.downsample_cf2cl(x)  # N C H W 2,20,224,320 -> N H W C,2,56,80,64->2,28,40,128->2,14,20,256->2,7,10,512 
        # features[2x256x56x80, 2x256x28x40, 2x256x14x20, 2x256x7x10]
        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            x[token_mask] = self.mask_token

        # if self.image_fusion:
        #     B,H,W,dim = x.shape
        #     x = x.view(B,H*W,dim).permute(1,0,2).contiguous()
        #     roi_features = self.embedder(roi_features).permute(1,0,2).contiguous()
        #     x = self.decoder(x, roi_features, memory_key_padding_mask=roi_mask)
        #     x = x.permute(1,0,2).view(B,H,W,dim).contiguous()

        for blk in self.att_blocks:
            x = blk(x)
        x = nhwC_2_nChw(x)  # N H W C -> N C H W 2,64,56,80

        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]

        # if self.image_fusion:
        #     roi_features = self.embedder(roi_features)
        #     shared_feature = th.cat([roi_features, x], dim=1)
        #     gates, moe_loss = self.moe_conv_layer(shared_feature) 
        #     gates = gates.view(-1, 2, 1, 1, 1) 
        #     moe_x = gates[:, 0] * roi_features + gates[:, 1] * x
        #     # weight = th.tensor([gates[:, 0].item(), gates[:, 1].item()])
        #     # if gates[:, 1].item() > 0.65 or gates[:, 0].item() > 0.65:
        #         # print(f'image weight is {gates[:, 0].item()}')
        #         # print(f'event weight is {gates[:, 1].item()}')
        #         # weight = True
        return x

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super(Mlp, self).__init__()
#         self.norm = nn.LayerNorm(in_features)
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.norm(x)
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x