import sys
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import transforms as T

import numpy as np

from mvsanywhere.utils.model_utils import get_model_class, load_model_inference

import os
from typing import List

class MVSA_Wrapped(nn.Module):
    def __init__(self):
        super().__init__()

        opts = MVSAOpts()
        self.opts = opts
        model_class_to_use = get_model_class(opts)
        model = load_model_inference(opts, model_class_to_use)
        model = model.eval()
        self.model = model
        self.name = "MVSAnywhere"

        self.use_refinement = True

    def forward(
        self,
        cur_data,
        src_data,
    ):

        outputs = self.model(
            phase="test",
            cur_data=cur_data,
            src_data=src_data,
            return_mask=True,
            num_refinement_steps=0,
        )
        return outputs


class MVSAOpts:
    def __init__(self):
        self.feature_volume_type = "view_agnostic_mlp_feature_volume"
        self.batch_size = 1
        self.val_batch_size = 1
        self.cost_volume_aggregation = "dot"
        self.cv_encoder_type = "vit_encoder"
        self.depth_decoder_name = "dpt"
        self.gpus = 1
        self.lr = 0.0001
        self.lr_da_encoder = 0.000005
        self.lr_da_decoder = 0.00005
        self.wd = 0.0001
        self.matching_encoder_type = "resnet"
        self.name = "simplerecon_model"
        self.num_sanity_val_steps = 0
        self.num_workers = 0
        self.precision = 16
        self.random_seed = 0
        self.image_width = 512
        self.image_height = 512
        self.val_image_width = 512
        self.val_image_height = 512
        self.model_type = "depth_model"

        self.da_weights_path = None
        self.matching_num_depth_bins = 64

        # Basic configuration
        self.random_seed = 0
        
        # Logging configuration
        self.log_dir = "output/tmp"
        self.notes = ""
        self.log_interval = 100
        self.val_interval = 1000
        self.val_batches = 100
        
        # Data configuration
        self.datasets = None
        self.val_datasets = None
        self.num_workers = 12
        self.model_num_views = 8
        self.num_images_in_tuple = 8
        self.shuffle_tuple = False
        self.test_keyframe_buffer_size = 30
        self.rotate_images = False
        
        # Hyperparameters
        self.max_steps = 220000
        self.precision = 16
        self.lr_steps = [70000, 80000]  # Default factory value
        
        # Model configuration
        self.resume = None
        self.load_weights_from_checkpoint = '/data/pug/mvsanywhere/src/mvsanywhere_hero.ckpt'
        self.lazy_load_weights_from_checkpoint = None
        self.image_encoder_name = "dinov2_vitb14"
        self.loss_type = "log_l1"
        self.matching_feature_dims = 16
        self.prediction_scale = 1.0
        self.prediction_num_scales = 5
        self.matching_scale = 0.25
        self.matching_num_depth_bins = 64
        self.min_matching_depth = 0.25
        self.max_matching_depth = 5.0
        self.da_weights_path = None
        
        # Inference configuration
        self.output_base_path = "results"
        self.run_fusion = True
        self.fuse_color = True
        self.fusion_max_depth = 5 # 3.0
        self.fusion_resolution = 0.02 # 0.02
        self.depth_fuser = "custom_open3d" # "ours"
        self.extended_neg_truncation = False
        self.single_debug_scan_id = None
        self.skip_frames = None
        self.skip_to_frame = None
        self.mask_pred_depth = False
        self.cache_depths = False
        self.fusion_use_raw_lowest_cost = False
        self.high_res_validation = False
        self.fast_cost_volume = True
        self.shift_world_origin = False
        self.save_for_regsplatfacto = False
        
        # Visualization configuration
        self.standard_fps = 30
        self.dump_depth_visualization = True
        self.use_precomputed_partial_meshes = False
        self.viz_render_width = 512
        self.viz_render_height = 512
        self.cam_marker_size = 0.7
        self.back_face_alpha = 0.5
        self.viz_fixed_min_max = False
        self.scan_parent_directory = None
        self.scan_name = None        
