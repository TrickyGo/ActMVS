import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from tools.operations import *
from tools.logger import TextColors
from .utils import (
    l1_loss_fc_mask,
    normal_tv_loss_fc,
    cons_loss_fc,
    WeightedSampler,
)

import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import copy



class GaussianMap:
    def __init__(self, cfg, device):
        self.device = device

        self._means = torch.empty(0, device=device)
        self._scales = torch.empty(0, device=device)
        self._rotations = torch.empty(0, device=device)
        self._opacities = torch.empty(0, device=device)
        self._harmonics = torch.empty(0, device=device)

        self.view_scores = torch.empty(0, device=device)
        self.view_supports = torch.empty(0, device=device)
        self.view_means = torch.empty((0, 3), device=self.device)

        self.training_performance = torch.tensor([], device=device)
        self.is_init = False

        self.gaussian_cfg = cfg["gaussian_map"] if cfg is not None else None
        self.use_view_distribution = False
        self.scene_near, self.scene_far = [0.001, 10.0]
        self.scale_factor = 0.01
        self.prune_interval = self.gaussian_cfg["prune_interval"] if cfg is not None else None
        self.optimization_steps = self.gaussian_cfg["optimization_steps"] if cfg is not None else None
        self.background_color = torch.tensor(
            [0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        ).to(self.device)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.is_depth_predictor_initialized = False
        self.output_dir = os.path.join(cfg["vis_dir"], 'render') if cfg is not None else None

    def update(self, training_data, dataframe_idx=-1, apply_depth_opt=False):
        self.apply_depth_opt = apply_depth_opt
        self.training_data = copy.deepcopy(training_data[:dataframe_idx+1])
        self.dataframe_idx = dataframe_idx
        dataframe = self.training_data[self.dataframe_idx]
        self.add_gaussians(dataframe)
        self.train()

    def train(self, steps=None):

        torch.cuda.empty_cache()
        self.init_training()
        training_sampler = self.get_sampler(self.training_data)
        iterations = self.optimization_steps if steps is None else steps

        for i in tqdm(
            range(iterations),
            desc=f" {TextColors.CYAN}Train Gaussian Map{TextColors.RESET}",
        ):
            [rgb_gts, depth_gts, extrinsics, intrinsics], frame_ids = (
                training_sampler.next_frames(self.training_performance, self.apply_depth_opt)
            )
            *_, h, w = rgb_gts.shape
            (
                rgb_preds,
                depth_preds,
                normal_preds,
                opacity_preds,
                d2n_preds,
                _,
                _,
                _,
                _,
            ) = GaussianRenderer(
                extrinsics,
                intrinsics,
                self.get_attr(),
                self.background_color,
                (self.scene_near, self.scene_far),
                (h, w),
                self.device,
            ).render_view_all(
                require_grad=True
            )

            mask_vis = opacity_preds.detach() > 1e-3
            mask_depth = depth_gts > 0.0

            rgb_loss = l1_loss_fc_mask(rgb_preds, rgb_gts, mask_vis)
            depth_loss = l1_loss_fc_mask(depth_preds, depth_gts, mask_depth)
            self.track_performance(rgb_loss, depth_loss, frame_ids)

            rgb_loss = rgb_loss.mean()
            depth_loss = depth_loss.mean()
            normal_cons_loss = normal_tv_loss_fc(normal_preds, depth_preds, mask_depth)
            consistency_loss = cons_loss_fc(normal_preds, d2n_preds)
            consistency_loss = (consistency_loss * mask_vis.long()).mean()

            total_loss = (
                rgb_loss
                + 0.8 * depth_loss
                + 0.1 * consistency_loss
                + 0.1 * normal_cons_loss
            )
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.post_processing()
        self.is_init = True

    def track_performance(self, rgb_loss, depth_loss, frame_ids):

        rgb_errs = torch.mean(rgb_loss, dim=[1, 2, 3])
        depth_errs = torch.mean(depth_loss, dim=[1, 2, 3])
        self.training_performance[frame_ids] = rgb_errs.detach() + depth_errs.detach()

    def post_processing(self):

        num_training_frame = len(self.training_data)
        require_prune = num_training_frame % self.prune_interval == 0

        if require_prune:
            extrinsics = torch.stack(
                [self.training_data[i]["extrinsic"] for i in range(num_training_frame)]
            )
            intrinsics = torch.stack(
                [self.training_data[i]["intrinsic"] for i in range(num_training_frame)]
            )
            rgb_gts = torch.stack(
                [self.training_data[i]["rgb"] for i in range(num_training_frame)]
            )
            depth_gts = torch.stack(
                [self.training_data[i]["depth"] for i in range(num_training_frame)]
            )

        else:
            extrinsics = self.training_data[self.dataframe_idx]["extrinsic"].unsqueeze(0)
            intrinsics = self.training_data[self.dataframe_idx]["intrinsic"].unsqueeze(0)
            rgb_gts = self.training_data[self.dataframe_idx]["rgb"].unsqueeze(0)
            depth_gts = self.training_data[self.dataframe_idx]["depth"].unsqueeze(0)

        *_, h, w = rgb_gts.shape
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            counts,
            _,
        ) = GaussianRenderer(
            extrinsics,
            intrinsics,
            self.get_attr(),
            self.background_color,
            (self.scene_near, self.scene_far),
            (h, w),
            self.device,
            render_masks=(depth_gts > 0.0).float(),
        ).render_view_all(require_importance=True, front_only=True)

        update_mask = counts[-1] >= 1.0
        self.view_supports += update_mask.float()

        if require_prune:
            vis_mask = (
                torch.sum(counts, dim=0) >= 1.0
            )
            self.prune(~vis_mask)



    def prune(self, prune_mask):
        prune_mask += self.get_opacities < 0.1
        prune_mask = prune_mask.bool()
        self._means = self._means[~prune_mask]
        self._scales = self._scales[~prune_mask]
        self._rotations = self._rotations[~prune_mask]
        self._opacities = self._opacities[~prune_mask]
        self._harmonics = self._harmonics[~prune_mask]
        self.view_scores = self.view_scores[~prune_mask]
        self.view_supports = self.view_supports[~prune_mask]
        self.view_means = self.view_means[~prune_mask]

        print(f"delete {torch.sum(prune_mask)} gaussians")

    def get_sampler(self, training_data):
        sampler = WeightedSampler(self.gaussian_cfg["sampler"], training_data)
        return sampler

    def init_training(self):
        self._means = nn.Parameter(self._means)
        self._scales = nn.Parameter(self._scales)
        self._rotations = nn.Parameter(self._rotations)
        self._opacities = nn.Parameter(self._opacities)
        self._harmonics = nn.Parameter(self._harmonics)
        l = [
            {
                "params": [self._means],
                "lr": self.gaussian_cfg["optimizer"]["mean_lr"],
                "name": "mean",
            },
            {
                "params": [self._scales],
                "lr": self.gaussian_cfg["optimizer"]["scale_lr"],
                "name": "scale",
            },
            {
                "params": [self._rotations],
                "lr": self.gaussian_cfg["optimizer"]["rotation_lr"],
                "name": "rotation",
            },
            {
                "params": [self._opacities],
                "lr": self.gaussian_cfg["optimizer"]["opacity_lr"],
                "name": "opacity",
            },
            {
                "params": [self._harmonics],
                "lr": self.gaussian_cfg["optimizer"]["harmonic_lr"],
                "name": "harmonic",
            },
        ]
        self.optimizer = torch.optim.Adam(l, eps=1e-15)



    def render_view(self, dataframe):
        rgb = dataframe["rgb"]
        intrinsic = dataframe["intrinsic"]
        extrinsic = dataframe["extrinsic"]

        _, H, W = rgb.shape
        point_num = H * W

        (
            rgb_pred,
            depth_pred,
            normal_pred,
            opacity_pred,
            _,
            confidence_pred,
            _,
            _,
            _,
        ) = GaussianRenderer(
            extrinsic.unsqueeze(0).to(self.device),
            intrinsic.unsqueeze(0).to(self.device),
            self.get_attr(),
            self.background_color,
            (self.scene_near, self.scene_far),
            (H, W),
            self.device,
        ).render_view_all()

        global_render_results = {
            "rgb": rgb_pred,
            "depth": depth_pred.squeeze(1),
            "opacity": opacity_pred.squeeze(1),
            "confidence": confidence_pred.squeeze(1),
            "normal": normal_pred,
        }
        return global_render_results



    def add_gaussians(self, dataframe):
        rgb = dataframe["rgb"]
        intrinsic = dataframe["intrinsic"]
        extrinsic = dataframe["extrinsic"]
        depth = dataframe["depth"]
        valid_mask = (depth > 0.0).view(-1)

        _, H, W = rgb.shape
        point_num = H * W


        if self.is_init:
            (
                rgb_pred,
                depth_pred,
                normal_pred,
                opacity_pred,
                _,
                confidence_pred,
                _,
                _,
                _,
            ) = GaussianRenderer(
                extrinsic.unsqueeze(0).to(self.device),
                intrinsic.unsqueeze(0).to(self.device),
                self.get_attr(),
                self.background_color,
                (self.scene_near, self.scene_far),
                (H, W),
                self.device,
            ).render_view_all()

            global_render_results = {
                "rgb": rgb_pred,
                "depth": depth_pred.squeeze(1),
                "opacity": opacity_pred.squeeze(1),
                "confidence": confidence_pred.squeeze(1),
                "normal": normal_pred,
            }


            select_mask = self.cal_mask(
                rgb.unsqueeze(0),
                depth.unsqueeze(0),
                global_render_results,
            )
            mask = select_mask.view(1,1,H,W)


            images = []
            H, W = rgb.shape[-2], rgb.shape[-1]
            rgb_img = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(rgb_img))
            depth_colored = tensor_to_colormap(depth.squeeze(), cmap="jet", vmin=0, vmax=10)
            images.append(depth_colored)
            mask_img = tensor_to_colormap(mask[0, 0].float() if select_mask.dim() == 4 else mask.squeeze().float(), 
                                        cmap="gray")
            images.append(mask_img)
            rgb_pred = global_render_results["rgb"][0]
            rgb_pred = torch.clamp(rgb_pred, 0.0, 1.0)
            rgb_pred_img = (rgb_pred.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(rgb_pred_img))
            depth_pred = global_render_results["depth"][0]
            depth_pred_img = tensor_to_colormap(depth_pred.squeeze(), cmap="jet", vmin=0, vmax=10)
            images.append(depth_pred_img)
            normal_pred = global_render_results["normal"][0]
            normal_img = (normal_pred.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255
            normal_img = normal_img.astype(np.uint8)
            images.append(Image.fromarray(normal_img))
            opacity_pred = global_render_results["opacity"][0]
            opacity_img = tensor_to_colormap(opacity_pred.squeeze(), cmap="viridis")
            images.append(opacity_img)

            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            combined = Image.new('RGB', (total_width, max_height))
            
            x_offset = 0
            for img in images:
                combined.paste(img, (x_offset, 0))
                x_offset += img.size[0]
            save_path = os.path.join(self.output_dir, f"gs_{self.dataframe_idx:04d}.png")
            combined.save(save_path)

        else:
            global_render_results = None
            depth = dataframe["depth"]


        depth_smooth = get_smooth_depth(depth.squeeze(0).cpu().numpy())
        depth_smooth = torch.tensor(depth_smooth, device=self.device).unsqueeze(0)

        xy_ray, _ = sample_image_grid((H, W), device=self.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        origins, directions = get_world_rays(xy_ray, extrinsic, intrinsic)
        pcd = (origins + directions * depth.view(-1, 1, 1)).squeeze(1)

        pcd_normals = torch.zeros(point_num, 3, device=self.device)
        pcd_normals[:, 2] = 1.0
        pcd_normals_cam = torch.zeros(point_num, 3, device=self.device)
        pcd_normals_cam[:, 2] = 1.0

        normals_cam = (
            depth2normal(
                depth_smooth, valid_mask.view(1, H, W), fov=(np.pi / 3, np.pi / 3)
            )
            .permute(1, 2, 0)
            .view(-1, 3)
        )
        valid_normal_mask = torch.sum(normals_cam**2, dim=-1) > 0.0
        valid_mask *= valid_normal_mask

        normals_world = torch.matmul(extrinsic[:3, :3], normals_cam.T).T
        pcd_normals_cam[valid_mask] = normals_cam[valid_mask]
        pcd_normals[valid_mask] = normals_world[valid_mask]

        directions_norm = torch.nn.functional.normalize(
            directions.squeeze(1), dim=1
        )
        cos_sim = torch.sum(directions_norm * pcd_normals, dim=-1)
        valid_normal_mask = cos_sim < -0.01
        valid_mask *= valid_normal_mask


        means_new = pcd
        rotations_new, _ = normal2rotation(pcd_normals)
        scales_new = torch.zeros_like(means_new, device=self.device)
        scales_new[:, -1] -= 1e10
        opacities_new = torch.zeros(point_num, device=self.device)
        harmonics_new = torch.zeros(point_num, 1, 3, device=self.device)
        harmonics_new[:, 0, :] = rgb.permute(1, 2, 0).view(-1, 3)

        view_scores_new = torch.zeros(point_num, device=self.device)
        view_supports_new = torch.zeros(point_num, device=self.device)
        view_means_new = torch.zeros((point_num, 3), device=self.device)

        nan_rotation_mask = torch.any(rotations_new.isnan(), dim=1)
        valid_mask *= ~nan_rotation_mask

        select_mask = self.cal_mask(
            rgb.unsqueeze(0),
            depth.unsqueeze(0),
            global_render_results,
        )
        select_mask = select_mask.to(self.device).squeeze(0) * valid_mask
        selected_idx = torch.nonzero(select_mask, as_tuple=False).flatten()

        select_mask_final = torch.zeros_like(select_mask, dtype=torch.bool)
        test_mask = torch.zeros(len(selected_idx), dtype=torch.bool)
        selected_pcd = pcd[select_mask]
        vf_idx = voxel_downsample(selected_pcd.to(self.device))
        test_mask[vf_idx] = True
        selected_idx = selected_idx[test_mask]
        select_mask_final[selected_idx] = True
        select_mask = select_mask_final

        self._means = torch.cat(
            (self._means.detach(), means_new.float()[select_mask]),
            dim=0,
        )
        self._scales = torch.cat(
            (self._scales.detach(), scales_new.float()[select_mask]),
            dim=0,
        )

        self._harmonics = torch.cat(
            (
                self._harmonics.detach(),
                harmonics_new.float()[select_mask],
            ),
            dim=0,
        )
        self._opacities = torch.cat(
            (
                self._opacities.detach(),
                opacities_new.float()[select_mask],
            ),
            dim=0,
        )
        self._rotations = torch.cat(
            (
                self._rotations.detach(),
                rotations_new.float()[select_mask],
            ),
            dim=0,
        )

        self.view_scores = torch.cat(
            (
                self.view_scores,
                view_scores_new.float()[select_mask],
            ),
            dim=0,
        )

        self.view_supports = torch.cat(
            (
                self.view_supports,
                view_supports_new.float()[select_mask],
            ),
            dim=0,
        )

        self.view_means = torch.cat(
            (
                self.view_means,
                view_means_new.float()[select_mask],
            ),
            dim=0,
        )

        self.training_performance = torch.cat(
            (self.training_performance, torch.tensor([10], device=self.device)), 0
        )

    def cal_mask(self, rgb_gt, depth_gt, pred):
        """
        get mask for spawning new gaussian primitives
        """
        v, _, h, w = rgb_gt.shape
        device = rgb_gt.device
        if pred is not None:
            opacity = pred["opacity"].to(device)
            mask = opacity < 0.8
        else:
            mask = torch.ones(v, h, w).to(device)

        return rearrange(mask.bool(), "v h w -> (v h w)")

    def save(self, save_path, index="final"):
        map_state = {
            "means": self._means.detach(),
            "scales": self._scales.detach(),
            "harmonics": self._harmonics.detach(),
            "opacities": self._opacities.detach(),
            "rotations": self._rotations.detach(),
            "view_scores": self.view_scores.detach(),
            "view_supports": self.view_supports.detach(),
            "view_means": self.view_means.detach(),
            "near": self.scene_near,
            "far": self.scene_far,
            "use_view_direction": self.use_view_distribution,
            "background_color": self.background_color,
            "scale_factor": self.scale_factor,
        }
        torch.save(map_state, f"{save_path}/map_{index}.th")

    def save_ply(self, save_path, index="final"):
        from plyfile import PlyData, PlyElement
        points = self._means.detach().cpu().numpy()        
        colors = self._harmonics.squeeze(1).detach().cpu().numpy() 

        vertex_data = np.empty(
            points.shape[0],
            dtype=[
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1")
            ]
        )

        vertex_data["x"] = points[:, 0].astype(np.float32)
        vertex_data["y"] = points[:, 1].astype(np.float32)
        vertex_data["z"] = points[:, 2].astype(np.float32)

        vertex_data["red"] = (np.clip(colors[:, 0], 0, 1) * 255).astype(np.uint8)
        vertex_data["green"] = (np.clip(colors[:, 1], 0, 1) * 255).astype(np.uint8)
        vertex_data["blue"] = (np.clip(colors[:, 2], 0, 1) * 255).astype(np.uint8)
        
  
        ply_element = PlyElement.describe(vertex_data, "vertex")
        PlyData([ply_element], text=False).write(f"{save_path}/map_{index}.ply")

    def load(self, model_path):
        map_state = torch.load(model_path)
        self._means = map_state["means"]
        self._scales = map_state["scales"]
        self._harmonics = map_state["harmonics"]
        self._opacities = map_state["opacities"]
        self._rotations = map_state["rotations"]
        self.view_scores = map_state["view_scores"]
        self.view_supports = map_state["view_supports"]
        self.view_means = map_state["view_means"]
        self.scene_near = map_state["near"]
        self.scene_far = map_state["far"]
        self.use_view_distribution = False
        self.background_color = torch.tensor(
            map_state["background_color"], dtype=torch.float32
        ).to(self.device)
        self.scale_factor = map_state["scale_factor"]
        self.is_init = True

    @property
    def get_means(self):
        return self._means

    @property
    def get_rotations(self):
        return self.rotation_activation(self._rotations)

    @property
    def get_scales(self):
        return torch.clamp(
            self.scale_factor * self.scaling_activation(self._scales), min=0, max=0.05
        )

    @property
    def get_opacities(self):
        return self.opacity_activation(self._opacities)

    @property
    def get_harmonics(self):
        return self._harmonics

    @property
    def get_confidences(self):
        confidences = torch.clamp(
            1 - 1 / torch.exp(self.view_supports), min=0, max=1
        )

        return confidences

    def get_attr(self):
        return (
            self.get_means,
            self.get_harmonics,
            self.get_opacities,
            self.get_confidences,
            self.get_scales,
            self.get_rotations,
        )


def tensor_to_colormap(tensor, cmap="viridis", vmin=None, vmax=None):
    """Convert 2D tensor to colormapped image"""
    if vmin is None:
        vmin = tensor.min()
    if vmax is None:
        vmax = tensor.max()
    normalized = (tensor - vmin) / max(vmax - vmin, 1e-8)
    normalized = normalized.clamp(0, 1).cpu().numpy()
    cm = plt.get_cmap(cmap)
    colored = (cm(normalized)[..., :3] * 255).astype(np.uint8) 
    return Image.fromarray(colored)
