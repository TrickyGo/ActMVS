import torch
import time

from tools.operations import *
from tools.logger import TextColors 
from .gaussian_map import GaussianMap
from .voxel_map import VoxelMap

import copy
import torch.nn.functional as F
from mapping.MVSA_wrapper import MVSA_Wrapped
from mvsanywhere.utils.generic_utils import imagenet_normalize
import os
import glob
import torch.optim as optim
import cv2


class Mapper:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.gaussian_map = None
        self.voxel_map = None

        self.MVS_model = MVSA_Wrapped()
        self.MVS_model = self.MVS_model.eval()

        self.apply_depth_opt = True
        self.max_src_frames = 8

        self.keyframes = []
        self.keyframe_frustum_mask = []
        self.kf_ref_dict = {}
        self.stage = 'wait_for_init_views'


    @property
    def current_map(self):
        return self.gaussian_map, self.voxel_map

    def load_recorder(self, recorder):
        print("\n ----------load mission recorder----------")
        self.recorder = recorder

    def load_simulator(self, simulator):
        print("\n ----------load simulator----------")
        self.simulator = simulator

    def load_planner(self, planner):
        print("\n ----------load planner----------")
        self.planner = planner

    def init_map(self):
        print("\n ----------initialize map----------")
        self.gaussian_map = GaussianMap(self.cfg, self.device)
        self.voxel_map = VoxelMap(self.cfg, self.simulator.bbox, self.device)
    
    def run(self):
        torch.cuda.empty_cache()
        self.init_map()
        frame_id = 0

        print(
            f"\n {TextColors.MAGENTA}----------Start Active Reconstruction----------{TextColors.RESET}"
        )
        while self.recorder is None or self.recorder.is_alive:

            print(
                f"\n {TextColors.MAGENTA}----------Step {frame_id+1}----------{TextColors.RESET}"
            )

            print(f"\n {TextColors.GREEN}-----Planning:{TextColors.RESET}")

            path_dataframes = self.get_new_path_dataframes()
            for dataframe in path_dataframes:
                for k, v in dataframe.items():
                    dataframe[k] = v.to(self.device)

                if frame_id > 0:
                    global_render_results = self.gaussian_map.render_view(dataframe)
                    dataframe["depth"] = global_render_results["depth"]
                else:
                    dataframe["depth"] = torch.zeros_like(dataframe['rgb'][0:1, :, :]).cuda()
                self.keyframe_frustum_mask += [self.voxel_map.get_frame_frustum_mask(dataframe)]
                self.keyframes += [dataframe]

            print(f"\n {TextColors.GREEN}-----Mapping:{TextColors.RESET}")

            print(f"planned views : {self.planner.planned_views_num}, keyframes :{ len(self.keyframes)}")

            if self.planner.planned_views_num == self.planner.init_views_num:
                self.stage = 'mapping_init_views'

            if self.stage == 'wait_for_init_views':
                pass
            elif self.stage == 'mapping_init_views':
                for idx in range(len(self.keyframes)):
                    dataframe = copy.deepcopy(self.keyframes[idx])
                    cur_kf = copy.deepcopy(self.keyframes[idx])
                    _, h, w = cur_kf["rgb"].shape

                    cur_kf["K"] = cur_kf["intrinsic"].clone() 
                    cur_kf["K"][0, :] *= w
                    cur_kf["K"][1, :] *= h
                    cur_kf["K"][:2, :] /= 4 

                    cur_data = {
                        "image_b3hw": imagenet_normalize(cur_kf["rgb"].unsqueeze(0)).cuda(),
                        "cam_T_world_b44": torch.inverse(cur_kf["extrinsic"]).unsqueeze(0),
                        "world_T_cam_b44": cur_kf["extrinsic"].unsqueeze(0),
                        "K_matching_b44": torch.cat([
                            torch.cat([cur_kf["K"], torch.zeros(3, 1, device=cur_kf["K"].device)], dim=1),
                            torch.tensor([[0, 0, 0, 1]], device=cur_kf["K"].device)
                        ]).unsqueeze(0),
                        "invK_matching_b44": torch.cat([
                            torch.cat([torch.inverse(cur_kf["K"]), torch.zeros(3, 1, device=cur_kf["K"].device)], dim=1),
                            torch.tensor([[0, 0, 0, 1]], device=cur_kf["K"].device)
                        ]).unsqueeze(0),                      
                    }

                    
                    src_idx_list = [src_idx for src_idx in list(range(len(self.keyframes))) if src_idx != idx][:16]
                    src_frames = [copy.deepcopy(self.keyframes[idx]) for idx in src_idx_list]
                    src_images = []
                    src_extrinsics = []

                    for kf in src_frames:
                        kf["K"] = kf["intrinsic"].clone()
                        kf["K"][0, :] *= w
                        kf["K"][1, :] *= h
                        kf["K"][:2, :] /= 4 
                        src_images.append(imagenet_normalize(kf["rgb"]))
                        src_extrinsics.append(kf["extrinsic"])

                    src_data = {
                        "image_b3hw": torch.stack(src_images).unsqueeze(0).cuda(),
                        "cam_T_world_b44": torch.inverse(torch.stack(src_extrinsics)).unsqueeze(0),
                        "world_T_cam_b44": torch.stack(src_extrinsics).unsqueeze(0),
                        "K_matching_b44": torch.stack([
                            torch.cat([
                                torch.cat([kf["K"], torch.zeros(3, 1, device=kf["K"].device)], dim=1),
                                torch.tensor([[0, 0, 0, 1]], device=kf["K"].device)
                            ]) for kf in src_frames
                        ]).unsqueeze(0).cuda(),
                        "invK_matching_b44": torch.stack([
                            torch.inverse(torch.cat([
                                torch.cat([kf["K"], torch.zeros(3, 1, device=kf["K"].device)], dim=1),
                                torch.tensor([[0, 0, 0, 1]], device=kf["K"].device)
                            ])) for kf in src_frames
                        ]).unsqueeze(0).cuda()
                    }


                    
                    with torch.no_grad():
                        outputs = self.MVS_model.forward(cur_data=cur_data, src_data=src_data)
                    predicted_depth = outputs["depth_pred_s0_b1hw"]
                    predicted_depth = F.interpolate(
                                            predicted_depth,
                                            size=(h, w),    
                                            mode='bilinear',    
                                            align_corners=False  
                                        ).squeeze(0)

                    dataframe["depth"] = predicted_depth
                    dataframe["rgb"] = dataframe["rgb"].cuda()
                    dataframe["depth"] = dataframe["depth"].cuda()
                    dataframe["intrinsic"] = dataframe["intrinsic"].cuda()
                    dataframe["extrinsic"] = dataframe["extrinsic"].cuda()
                    
                    self.keyframes[idx] = dataframe
                    self.kf_ref_dict[idx] = src_idx_list

                    for dataframe in self.keyframes:
                        if "opt_depth" not in dataframe.keys():
                            dataframe["opt_depth"] = dataframe["depth"]
                    self.depth_opt(idx)

                    self.gaussian_map.update(self.keyframes, dataframe_idx=idx, apply_depth_opt=self.apply_depth_opt)

                    self.voxel_map.update(dataframe)

                    frame_id += 1

                self.stage = 'incremental_mapping'

            elif self.stage == 'incremental_mapping':
                for idx in range(len(self.keyframes)-len(path_dataframes), len(self.keyframes)):
                    t_mapper_start = time.time()
                    dataframe = copy.deepcopy(self.keyframes[idx])

                    cur_kf = copy.deepcopy(self.keyframes[idx])
                    _, h, w = cur_kf["rgb"].shape

                    cur_kf["K"] = cur_kf["intrinsic"].clone() 
                    cur_kf["K"][0, :] *= w
                    cur_kf["K"][1, :] *= h
                    cur_kf["K"][:2, :] /= 4 

                    cur_data = {
                        "image_b3hw": imagenet_normalize(cur_kf["rgb"].unsqueeze(0)).cuda(),
                        "cam_T_world_b44": torch.inverse(cur_kf["extrinsic"]).unsqueeze(0),
                        "world_T_cam_b44": cur_kf["extrinsic"].unsqueeze(0),
                        "K_matching_b44": torch.cat([
                            torch.cat([cur_kf["K"], torch.zeros(3, 1, device=cur_kf["K"].device)], dim=1),
                            torch.tensor([[0, 0, 0, 1]], device=cur_kf["K"].device)
                        ]).unsqueeze(0),
                        "invK_matching_b44": torch.cat([
                            torch.cat([torch.inverse(cur_kf["K"]), torch.zeros(3, 1, device=cur_kf["K"].device)], dim=1),
                            torch.tensor([[0, 0, 0, 1]], device=cur_kf["K"].device)
                        ]).unsqueeze(0), 
                    }

                    searched_src_frames_idx_list = self.select_ref_pose(idx, self.max_src_frames)

                    if len(searched_src_frames_idx_list) < 2:
                        self.gaussian_map.training_performance = torch.cat(
                            (self.gaussian_map.training_performance, torch.tensor([0.01], device=self.gaussian_map.device)), 0
                        )
                        continue 
                    
                    src_frames = [copy.deepcopy(self.keyframes[i]) for i in searched_src_frames_idx_list]
                    src_images = []
                    src_extrinsics = []

                    for kf in src_frames:
                        kf["K"] = kf["intrinsic"].clone()
                        kf["K"][0, :] *= w
                        kf["K"][1, :] *= h
                        kf["K"][:2, :] /= 4
                        src_images.append(imagenet_normalize(kf["rgb"]))
                        src_extrinsics.append(kf["extrinsic"])

                    src_data = {
                        "image_b3hw": torch.stack(src_images).unsqueeze(0).cuda(),
                        "cam_T_world_b44": torch.inverse(torch.stack(src_extrinsics)).unsqueeze(0),
                        "world_T_cam_b44": torch.stack(src_extrinsics).unsqueeze(0),
                        "K_matching_b44": torch.stack([
                            torch.cat([
                                torch.cat([kf["K"], torch.zeros(3, 1, device=kf["K"].device)], dim=1),
                                torch.tensor([[0, 0, 0, 1]], device=kf["K"].device)
                            ]) for kf in src_frames
                        ]).unsqueeze(0).cuda(),
                        "invK_matching_b44": torch.stack([
                            torch.inverse(torch.cat([
                                torch.cat([kf["K"], torch.zeros(3, 1, device=kf["K"].device)], dim=1),
                                torch.tensor([[0, 0, 0, 1]], device=kf["K"].device)
                            ])) for kf in src_frames
                        ]).unsqueeze(0).cuda()
                    }

                    with torch.no_grad():
                        outputs = self.MVS_model.forward(cur_data=cur_data, src_data=src_data)
                    
                    predicted_depth = outputs["depth_pred_s0_b1hw"]

                    predicted_depth = F.interpolate(
                                            predicted_depth,
                                            size=(h, w),    
                                            mode='bilinear',    
                                            align_corners=False  
                                        ).squeeze(0)

                    dataframe["depth"] = predicted_depth

                    self.kf_ref_dict[idx] = searched_src_frames_idx_list

                    self.keyframes[idx] = dataframe

                    if self.apply_depth_opt:
                        for dataframe in self.keyframes:
                            if "opt_depth" not in dataframe.keys():
                                dataframe["opt_depth"] = dataframe["depth"]
                        self.depth_opt(idx)
                    self.gaussian_map.update(self.keyframes, dataframe_idx=idx, apply_depth_opt=self.apply_depth_opt)

                    self.voxel_map.update(dataframe)


                    t_mapper = time.time() - t_mapper_start
                    frame_id += 1

                    if self.recorder is not None:
                        self.recorder.update_time("mapping", t_mapper)
                        self.recorder.log()
                        self.recorder.save_dataframe(dataframe)
                        if self.recorder.require_record:
                            self.recorder.save_map(self.gaussian_map, f"{frame_id:03}")

        render_vis_dir = os.path.join(self.cfg["vis_dir"], 'render')
        video_path = os.path.join(self.cfg["vis_dir"], "vis_video.mp4")
        create_video(render_vis_dir, video_path)

        print(
            f"\n {TextColors.MAGENTA}----------Finish Reconstruction Mission----------{TextColors.RESET}"
        )

    def _get_L2_dis(self, ex1, ex2):
        T1 = ex1[:3, 3].cpu().numpy()
        T2 = ex2[:3, 3].cpu().numpy()
        distance = np.linalg.norm(T1 - T2)
        return distance

    def select_ref_pose(self, cur_idx, topk):
        d_min = 0.1
        d_max = 0.5
        angle_min = np.deg2rad(0)
        angle_max = np.deg2rad(25)

        count_record = []
        for index in range(len(self.keyframes)):
            if index != cur_idx:
                score = 0.0

                cur_position = self.keyframes[cur_idx]['extrinsic'][:3, 3].cpu().numpy()
                cur_direction = self.keyframes[cur_idx]['extrinsic'][:3, 2].cpu().numpy()

                ref_position = self.keyframes[index]['extrinsic'][:3, 3].cpu().numpy()
                ref_direction = self.keyframes[index]['extrinsic'][:3, 2].cpu().numpy()

                distance = np.linalg.norm(cur_position - ref_position)
                cos_angle = np.dot(cur_direction, ref_direction)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                if distance < d_min or distance > d_max:
                    continue
                
                if angle < angle_min or angle > angle_max:
                    continue
                
                common_mask = torch.logical_and(self.keyframe_frustum_mask[index], self.keyframe_frustum_mask[cur_idx])
                common_voxel_num = torch.sum(common_mask)
                distance = self._get_L2_dis(self.keyframes[index]['extrinsic'], self.keyframes[cur_idx]['extrinsic'])
                score = common_voxel_num / 1000 + distance / 5 if common_voxel_num >= 50 else 0
                
                count_record.append((index, score))
        
        max_score = sorted(count_record, key=lambda x: x[1], reverse=True)
        top_indexes = [item[0] for item in max_score[:topk]]

        return top_indexes


    def get_new_path_dataframes(self):
        path = self.planner.plan(self.current_map, self.recorder)

        path_dataframes = []
        for view in path[1:]:
            dataframe = self.simulator.simulate(view)
            path_dataframes += [dataframe]
        path_dataframes = path_dataframes[::5]


        return path_dataframes

    def depth_opt(self, idx, num_iters=4, lr=1e-3):
        huber_delta = 0.1
        tv_weight = 1e-4
        opt_ref_views_num = 3
        self.kf_ref_dict[idx] = self.kf_ref_dict[idx][:opt_ref_views_num]

        keyframes = self.keyframes
        n_frames = len(keyframes)
        
        depth_params = []
        for i in range(n_frames):
            depth = keyframes[i]["depth"].detach().clone().requires_grad_(True)
            depth_params.append(depth)

        optimizer = optim.Adam(depth_params, lr=lr)
        
        _, h, w = keyframes[0]["depth"].shape
        
        T_world_to_cam = []
        for frame in keyframes:
            T_world_to_cam.append(torch.inverse(frame["extrinsic"]))
        
        device = self.device
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        for _ in range(num_iters):
            optimizer.zero_grad()
            total_loss = 0
            total_pairs = 0

            i = idx
            depth_cur = depth_params[i]
            K_cur = keyframes[i]["intrinsic"].clone()
            T_cur_to_world = keyframes[i]["extrinsic"]
            keyframes[idx]["depth_mask"] = torch.ones_like(depth_cur)

            K_cur[0, :] *= w
            K_cur[1, :] *= h
            
            for j in self.kf_ref_dict[i]:
                depth_ref = depth_params[j]
                K_ref = keyframes[j]["intrinsic"].clone()
                
                K_ref[0, :] *= w
                K_ref[1, :] *= h
                
                T_cur_to_ref = T_world_to_cam[j] @ T_cur_to_world
                
                points_cur = torch.zeros(1, 3, h, w, device=device)
                points_cur[:, 0, :, :] = (x_coords - K_cur[0, 2]) * depth_cur / K_cur[0, 0]
                points_cur[:, 1, :, :] = (y_coords - K_cur[1, 2]) * depth_cur / K_cur[1, 1]
                points_cur[:, 2, :, :] = depth_cur
                
                points_cur_homo = torch.cat([points_cur, torch.ones(1, 1, h, w, device=device)], dim=1)
                
                T_cur_to_ref_exp = T_cur_to_ref.reshape(1, 4, 4)
                points_ref_homo = torch.einsum('bij,bjhw->bihw', T_cur_to_ref_exp, points_cur_homo)
                points_ref = points_ref_homo[:, :3, :, :]
                
                z_ref = points_ref[:, 2, :, :].clamp(min=1e-6)
                x_proj = points_ref[:, 0, :, :] / z_ref
                y_proj = points_ref[:, 1, :, :] / z_ref
                
                u_proj = x_proj * K_ref[0, 0] + K_ref[0, 2]
                v_proj = y_proj * K_ref[1, 1] + K_ref[1, 2]
                
                grid_ref = torch.stack([u_proj, v_proj], dim=-1)
                grid_ref[..., 0] = 2.0 * grid_ref[..., 0] / (w - 1) - 1.0
                grid_ref[..., 1] = 2.0 * grid_ref[..., 1] / (h - 1) - 1.0
                
                depth_ref_sampled = F.grid_sample(
                    depth_ref.unsqueeze(0),
                    grid_ref,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                ).squeeze(0)
                
                depth_cur_in_ref = z_ref.squeeze(0) 

                log_depth_cur_in_ref = torch.log(depth_cur_in_ref.clamp(min=1e-2))
                log_depth_ref_sampled = torch.log(depth_ref_sampled.clamp(min=1e-2))

                diff = F.huber_loss(
                    log_depth_ref_sampled, 
                    log_depth_cur_in_ref, 
                    delta=huber_delta, 
                    reduction='none'
                )
                z_ref = points_ref[:, 2, :, :].clone()
                occlusion_mask = z_ref < (depth_ref_sampled + 0.1)

                in_bounds = (u_proj > 1.0) & (u_proj < w-1) & (v_proj > 1.0) & (v_proj < h-1)
                valid_mask = (depth_ref_sampled > 0) & (depth_cur_in_ref > 0) & in_bounds & occlusion_mask
                keyframes[i]["depth_mask"] *= valid_mask
                
                if valid_mask.any():
                    pair_loss = (diff * valid_mask.float()).sum() / valid_mask.sum()
                    total_loss += pair_loss
                    total_pairs += 1
                    

            if tv_weight > 0:
                smooth_loss = tv_loss(depth_params[i], valid_mask)
                total_loss += tv_weight * smooth_loss
            else:
                smooth_loss = 0

            if total_pairs > 0:
                avg_loss = total_loss / total_pairs
                avg_loss.backward()
                optimizer.step()
    
        self.keyframes[idx]["opt_depth"] = depth_params[idx].detach().clone() * keyframes[idx]["depth_mask"] + self.keyframes[idx]["opt_depth"] * (1 - keyframes[idx]["depth_mask"])
        

def tv_loss(depth_map, mask):
    diff_x = depth_map[..., :-1] - depth_map[..., 1:]
    diff_y = depth_map[..., :-1, :] - depth_map[..., 1:, :]
    
    loss_x = (mask[..., :-1] * mask[..., 1:] * torch.abs(diff_x)).sum()
    loss_y = (mask[..., :-1, :] * mask[..., 1:, :] * torch.abs(diff_y)).sum()
    
    valid_pixels = mask.sum()
    return (loss_x + loss_y) / (valid_pixels + 1e-6) if valid_pixels > 0 else 0

def create_video(image_dir, output_path, start_idx=0, fps=30):
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))[start_idx:]
    if not images:
        return
    
    img = cv2.imread(images[0])
    h, w, _ = img.shape
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img_path in images:
        video.write(cv2.imread(img_path))
    video.release()
