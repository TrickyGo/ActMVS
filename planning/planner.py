import numpy as np
import torch
import time

from .utils import (
    PathPlanner,
    cal_flight_time,
    inplace_rotation,
    wp2path,
)
from tools.logger import TextColors

class Planner:
    def __init__(self, cfg, device):
        self.device = device
        self.pitch_angle = None
        self.robot_size = cfg["planner"]["robot_size"]
        self.radius = cfg["planner"]["radius"]
        self.flight_speed = 1.0
        self.pose = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).type(torch.float32)

        self.path_planner = PathPlanner()
        self.path_length_factor = 0.0
        self.sample_num = cfg["planner"]["sample_num"]

        self.init_pose = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).type(torch.float32)
        self.stage = 'init'
        self.init_views_type = 'h_move'
        self.init_views_num = 4
        self.planned_views_num = 0 
        self.history_poses = []

    def plan(self, map_state, recorder):
        _, voxel_map = map_state
        t_planning = 0

        if self.stage == 'init' :
            nbv = torch.eye(4)
            nbv[:3, :3] = self.init_pose[:3, :3]
            nbv_index = voxel_map.xyz_2_index(self.init_pose[:3, 3])
            nbv_xyz = voxel_map.index_2_xyz([nbv_index])[0].cpu()
            nbv[:3, 3] = nbv_xyz

            total_distance = 1
            distance = total_distance * self.planned_views_num / self.init_views_num
            x = 0.0
            y = distance
            z = 0.0
            delta_nbv_xyz = torch.tensor([x, y, z]).float()
            nbv_xyz += delta_nbv_xyz
            nbv[:3, 3] += delta_nbv_xyz

            waypoints = torch.stack([self.pose[:3, 3], nbv_xyz])


            if self.planned_views_num == self.init_views_num:
                self.stage = 'incremental'

        else:
            t_sampling_start = time.time()
            robot_space = self.get_robot_space(voxel_map)
            voxel_map.update_graph(robot_space)

            total_candidates = self.generate_random_candidates(
                voxel_map, self.sample_num
            )
            t_planning += time.time() - t_sampling_start

            utility_list, t_utility = self.cal_utility(total_candidates)
            t_planning += t_utility

            t_path_start = time.time()
            wp_list, wp_length_list = self.path_planner.search_goal(
                self.pose[:3, 3].numpy(),
                total_candidates[:, :3, 3].numpy(),
                voxel_map,
            )
            t_planning += time.time() - t_path_start

            score_list = self.cal_view_scores(utility_list, wp_length_list)
            nbv_id = torch.argmax(score_list)
            nbv = total_candidates[nbv_id]
            wp_length = wp_length_list[nbv_id]

            if np.isinf(wp_length):
                print("inf path length!!!!!!!")
                assert 0

            wp_indices = wp_list[nbv_id]
            waypoints = voxel_map.index_2_xyz(wp_indices).cpu()

        camera_path, path_length = wp2path(
            self.pose[:3, :3],
            nbv[:3, :3],
            waypoints,
        )

        self.pose = nbv
        self.history_poses.append(nbv.clone().detach())
        self.planned_views_num += 1

        if recorder is not None:
            t_flight = cal_flight_time(path_length, flight_speed=self.flight_speed)
            recorder.update_time("planning", t_planning)
            recorder.update_time("flight", t_flight)

        return camera_path

    def generate_random_candidates(self, voxel_map, num):
        """
        generate random view candidates around current pose
        """

        voxel_centers = voxel_map.voxel_centers.cpu().numpy()
        free_mask = voxel_map.free_mask_w_margin.cpu().numpy()

        range_from_start = np.linalg.norm(
            voxel_centers - self.pose[:3, 3].numpy(), axis=1
        )
        within_range = range_from_start <= self.radius
        valid_mask = free_mask & within_range
        valid_centers = voxel_centers[valid_mask]
        random_indices = np.random.choice(len(valid_centers), size=num)
        view_positions = valid_centers[random_indices]
        candidates = inplace_rotation(
            view_positions, pitch_angle=self.pitch_angle, num=num
        )
        return candidates


    def get_robot_space(self, voxel_map):
        range_from_start = torch.linalg.norm(
            voxel_map.voxel_centers - self.pose[:3, 3].unsqueeze(0).to(self.device),
            dim=1,
        )
        robot_space = range_from_start < self.robot_size
        return robot_space

    def cal_view_scores(self, view_utilities, path_lengths):
        """
        calculate the score of each viewpoint based on its utility and travel cost
        """

        path_lengths = torch.tensor(path_lengths)
        valid_candidate_mask = ~torch.isinf(path_lengths)

        path_lengths = path_lengths / torch.sum(path_lengths[valid_candidate_mask])
        path_lengths[~valid_candidate_mask] = 10000000

        view_utilities = view_utilities / torch.sum(view_utilities)
        view_utilities[torch.isnan(view_utilities)] = 0
        if torch.all(view_utilities == 0):
            view_scores = torch.rand_like(view_utilities)
        else:
            view_scores = view_utilities - self.path_length_factor * path_lengths
        return view_scores

    @torch.no_grad
    def cal_utility(self, candidates):

        print(f" {TextColors.CYAN}Evaluate View Candidates{TextColors.RESET}")
        utility_list = torch.rand(len(candidates))
        return utility_list, 0

