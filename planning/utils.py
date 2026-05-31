import numpy as np
import heapq
from scipy.special import comb

from tools.operations import *


def cal_flight_time(path_length, flight_speed):
    return path_length / flight_speed


def inplace_rotation(point, pitch_angle=None, num=1):
    Ts = repeat(np.eye(4), "h w -> n h w", n=num)
    Ts[:, :3, 3] = point
    Ts[:, :3, :3] = random_rotation(num, pitch_angle)
    return torch.tensor(Ts).type(torch.float32)


class PathPlanner:
    def __init__(self):
        pass

    def final_output(self, goal_indices, paths, travel_distances):
        path_list = []
        travel_distance_list = []
        for goal_index in goal_indices:
            if tuple(goal_index) in paths:
                path_list.append(paths[tuple(goal_index)])
                travel_distance_list.append(travel_distances[tuple(goal_index)])
            else:
                path_list.append([])
                travel_distance_list.append(float("inf"))

        return path_list, travel_distance_list

    def search_goal(self, start, goals, voxel_map):
        size = voxel_map.size.cpu().numpy()
        dim = voxel_map.dim.cpu().numpy()
        bbox = voxel_map.bbox.cpu().numpy()
        voxel_centers = voxel_map.voxel_centers.cpu().numpy().reshape((*dim, 3))
        graph = voxel_map.graph.dense_graph

        start_index = tuple(np.floor((start - bbox[0]) / size).astype(int))
        goal_indices = np.array(
            [np.floor((goal - bbox[0]) / size).astype(int) for goal in goals]
        )

        distances = {node: float("inf") for node in graph}
        distances[start_index] = 0
        priority_queue = [(0, start_index)]
        parents = {start_index: None}

        remaining_goals = set(
            [tuple(goal) for goal in goal_indices if tuple(goal) in graph]
        )
        paths = {tuple(goal): [] for goal in remaining_goals}
        travel_distances = {tuple(goal): float("inf") for goal in remaining_goals}

        def heuristic(current_voxel):
            current_voxel_center = voxel_centers[tuple(current_voxel)]
            h = np.min(np.linalg.norm(goals - current_voxel_center, axis=1))
            return h

        while priority_queue and remaining_goals:
            current_f_score, current_node = heapq.heappop(priority_queue)

            if tuple(current_node) in remaining_goals:
                remaining_goals.remove(tuple(current_node))
                path = []
                node = current_node
                while node is not None:
                    path.append(node)
                    node = parents.get(node)
                path.reverse()
                paths[tuple(current_node)] = path
                travel_distances[tuple(current_node)] = distances[current_node]

                if not remaining_goals:
                    break

            for neighbor, weight in graph[current_node]:
                g_score = distances[current_node] + weight

                if g_score < distances[neighbor]:
                    distances[neighbor] = g_score
                    parents[neighbor] = current_node

                    f_score = g_score + heuristic(neighbor)
                    heapq.heappush(priority_queue, (f_score, neighbor))

        path_list, travel_distance_list = self.final_output(
            goal_indices, paths, travel_distances
        )
        return path_list, travel_distance_list

def rotation_from_z_batch(z_axis_batch):
    z_axis_batch = z_axis_batch / torch.norm(z_axis_batch, dim=-1, keepdim=True)
    batch_size = z_axis_batch.shape[0]
    y_axis = torch.tensor([0.0, 0.0, -1.0], device=z_axis_batch.device).expand(
        batch_size, -1
    )

    is_collinear = torch.all(
        torch.isclose(z_axis_batch, y_axis, atol=1e-3), dim=1
    ) | torch.all(torch.isclose(z_axis_batch, -y_axis, atol=1e-3), dim=1)

    x_axis = torch.where(
        is_collinear.unsqueeze(-1),
        torch.tensor([1.0, 0.0, 0.0], device=z_axis_batch.device).expand(
            batch_size, -1
        ),
        torch.cross(y_axis, z_axis_batch, dim=-1),
    )

    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)

    y_axis_new = torch.cross(z_axis_batch, x_axis, dim=-1)
    y_axis_new = y_axis_new / torch.norm(y_axis_new, dim=-1, keepdim=True)

    rotation_matrix = torch.stack((x_axis, y_axis_new, z_axis_batch), dim=-1)
    return rotation_matrix


def bezier_curve(control_points, num_points=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, len(control_points[0])))

    for i in range(n + 1):
        curve += np.outer(comb(n, i) * (t**i) * ((1 - t) ** (n - i)), control_points[i])

    return curve


def angle_between_vec(v1, v2):
    v1 = v1 / v1.norm(p=2, dim=-1, keepdim=True)
    v2 = v2 / v2.norm(p=2, dim=-1, keepdim=True)

    dot = torch.sum(v1 * v2, dim=-1)

    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)
    return theta


def slerp(v1, v2, t):
    theta_0 = angle_between_vec(v1, v2)
    sin_theta_0 = torch.sin(theta_0)

    if theta_0 < 1e-3:
        return v2.repeat(len(t), 1)

    sin_t_theta = torch.sin(t.unsqueeze(-1) * theta_0)
    sin_1_minus_t_theta = torch.sin(
        (1 - t.unsqueeze(-1)) * theta_0
    )

    v_interpolated = (
        sin_1_minus_t_theta * v1 + sin_t_theta * v2
    ) / sin_theta_0.unsqueeze(-1)

    return v_interpolated / v_interpolated.norm(p=2, dim=-1, keepdim=True)


def wp2path(
    start_rotation,
    goal_rotation,
    waypoints,
    distance_thre=0.05,
    angle_thre=0.1,
):
    start_view_direction = start_rotation[:, 2]
    goal_view_direction = goal_rotation[:, 2]
    angle_distance = angle_between_vec(start_view_direction, goal_view_direction)
    num_sample_angle = torch.ceil(angle_distance / angle_thre).long()

    if len(waypoints) == 1:
        path_length = torch.tensor(0)
        num_sample = num_sample_angle
        interpolated_positions = torch.tensor(waypoints[-1]).repeat(num_sample, 1)
    else:
        diffs = torch.tensor(waypoints[1:] - waypoints[:-1])
        path_length = torch.sum(torch.norm(diffs, dim=1))
        num_sample_xyz = torch.ceil(path_length / distance_thre).long()
        num_sample = max(num_sample_xyz, num_sample_angle)
        interpolated_positions = bezier_curve(waypoints, num_points=num_sample)

    t = torch.linspace(0, 1, num_sample)
    interpolated_view_directions = slerp(start_view_direction, goal_view_direction, t)
    interpolated_rotations = rotation_from_z_batch(interpolated_view_directions)

    path = torch.eye(4).repeat(num_sample, 1, 1)
    path[:, :3, 3] = torch.tensor(interpolated_positions)
    path[:, :3, :3] = torch.tensor(interpolated_rotations)

    return path, path_length.item()
