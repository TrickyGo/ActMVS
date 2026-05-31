import torch
import torch.nn.functional as F
import numpy as np


def cons_loss_fc(normals, depth_normals):
    cos = torch.sum(normals * depth_normals, 1)
    return 1 - cos


def normal_tv_loss_fc(normals, depths, mask, sigma=0.3):
    normal_diff_norms = central_diff(normals)
    depth_diff_norms = central_diff(depths.detach())

    depth_mask = (depth_diff_norms <= 0.0001).float()
    weights = torch.exp(-normal_diff_norms / (2 * sigma**2))

    loss = torch.mean(depth_mask * weights * normal_diff_norms * mask)
    return loss


def central_diff(map):
    shift_left = map[:, :, :, :-1] - map[:, :, :, 1:]
    shift_right = map[:, :, :, 1:] - map[:, :, :, :-1]
    shift_up = map[:, :, :-1, :] - map[:, :, 1:, :]
    shift_down = map[:, :, 1:, :] - map[:, :, :-1, :]

    pad = (0, 1, 0, 0)
    shift_left = F.pad(shift_left, pad, mode="constant", value=0)
    pad = (1, 0, 0, 0)
    shift_right = F.pad(shift_right, pad, mode="constant", value=0)
    pad = (0, 0, 0, 1)
    shift_up = F.pad(shift_up, pad, mode="constant", value=0)
    pad = (0, 0, 1, 0)
    shift_down = F.pad(shift_down, pad, mode="constant", value=0)
    diffs = torch.stack(
        [shift_left, shift_right, shift_up, shift_down], dim=2
    )

    diff_norms = torch.sum(diffs**2, dim=1)
    return diff_norms


def l1_loss_fc_mask(network_output, gt, mask):
    return torch.abs((network_output - gt) * mask)


class WeightedSampler:
    def __init__(self, cfg, dataframes):
        active_size = min(cfg["active_size"], len(dataframes))
        batch_size = cfg["batch_size"]
        self.dataframes = dataframes
        ids = range(len(self.dataframes))
        self.random_num = batch_size - active_size
        self.active_ids = np.array(ids[-active_size:])
        self.random_ids_all = np.array(ids[:-active_size])
        self.selected_num = min(len(self.random_ids_all), self.random_num)
        self.v = (
            len(self.active_ids) + self.selected_num
        )

    def next_frames(self, weight, apply_depth_opt=False):
        selected_ids = self.active_ids.copy()
        if self.selected_num > 0:
            weight = weight[self.random_ids_all]
            weight /= torch.sum(weight)
            indices = np.random.choice(
                self.random_ids_all,
                size=self.selected_num,
                p=weight.cpu().numpy(),
                replace=False,
            )
            ids = self.random_ids_all[indices]
            selected_ids = np.append(selected_ids, ids)

        rgbs = torch.stack([self.dataframes[i]["rgb"] for i in selected_ids])
        if apply_depth_opt:
            depths = torch.stack([self.dataframes[i]["opt_depth"] for i in selected_ids])
        else:
            depths = torch.stack([self.dataframes[i]["depth"] for i in selected_ids])
        extrinsics = torch.stack(
            [self.dataframes[i]["extrinsic"] for i in selected_ids]
        )
        intrinsics = torch.stack(
            [self.dataframes[i]["intrinsic"] for i in selected_ids]
        )
        return [rgbs, depths, extrinsics, intrinsics], selected_ids


def cal_distance(points, origin):
    return torch.sqrt(torch.sum((points - origin) ** 2, dim=-1) + 1e-8)
