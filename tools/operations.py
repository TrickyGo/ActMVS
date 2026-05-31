from jaxtyping import Float, Int64, Bool
import torch
from torch import Tensor
import numpy as np
import cv2
import torchvision.transforms as tf
from scipy.spatial.transform import Rotation as R
from einops import rearrange, repeat
import copy
import torchvision.transforms as tf
from math import isqrt, tan
from einops import einsum, rearrange, reduce, repeat
from torch.functional import norm
import torch.nn.functional as F
import math


from diff_gaussian_rasterization_2d import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


opencv_rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def random_rotation(n, pitch_angle, opencv=True):
    points = np.random.randn(n, 3)  # Sample from normal distribution
    points = points / np.clip(np.linalg.norm(points, axis=1, keepdims=True), 1e-8, None)

    z_rot = np.zeros(n)

    if pitch_angle is None:
        x_rot = np.arcsin(points[:, 2])
    else:
        x_rot = np.ones(n) * pitch_angle

    y_rot = np.arctan2(points[:, 1], points[:, 0])

    eulers = np.stack((z_rot, x_rot, y_rot), axis=-1)
    rotation_matrix = R.from_euler("zxy", eulers).as_matrix()
    if opencv:
        rotation_matrix = opencv_rotation @ rotation_matrix
    return rotation_matrix



def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def get_smooth_depth(depth, tolerance=0.5):
    invalid_mask = depth < 0.0
    valid_depth_image = np.copy(depth)
    valid_depth_image[invalid_mask] = np.nan
    filtered_depth = cv2.bilateralFilter(
        np.nan_to_num(valid_depth_image), 15, tolerance, 20
    )
    filtered_depth[invalid_mask] = -1.0
    return filtered_depth


def depth2normal(depth, mask, fov):
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(
        torch.arange(0, shape[0], device=device, dtype=torch.float32),
        torch.arange(0, shape[1], device=device, dtype=torch.float32),
        torch.arange(0, shape[2], device=device, dtype=torch.float32),
        indexing="ij",
    )
    p = torch.cat([w, h], axis=-1)

    p[..., 0:1] -= 0.5 * shape[1]
    p[..., 1:2] -= 0.5 * shape[0]
    p *= camD
    K00 = fov2focal(fov[0], shape[0])
    K11 = fov2focal(fov[1], shape[1])
    K = torch.tensor([K00, 0, 0, K11], device=device).reshape([2, 2])
    Kinv = torch.inverse(K)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    p_padded = torch.nn.functional.pad(
        camPos[None], [0, 0, 1, 1, 1, 1], mode="replicate"
    )
    mask_padded = torch.nn.functional.pad(
        mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode="replicate"
    ).to(torch.bool)

    p_c = p_padded[:, 1:-1, 1:-1, :] * mask_padded[:, 1:-1, 1:-1, :]
    p_u = (p_padded[:, :-2, 1:-1, :] - p_c) * mask_padded[:, :-2, 1:-1, :]
    p_l = (p_padded[:, 1:-1, :-2, :] - p_c) * mask_padded[:, 1:-1, :-2, :]
    p_b = (p_padded[:, 2:, 1:-1, :] - p_c) * mask_padded[:, 2:, 1:-1, :]
    p_r = (p_padded[:, 1:-1, 2:, :] - p_c) * mask_padded[:, 1:-1, 2:, :]

    n_ul = torch.cross(p_u, p_l)
    n_ur = torch.cross(p_r, p_u)
    n_br = torch.cross(p_b, p_r)
    n_bl = torch.cross(p_l, p_b)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]

    n = torch.nn.functional.normalize(n, dim=-1)

    n = (n * mask).permute([2, 0, 1])
    return n




def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def project_camera_space(
    points: Float[Tensor, "*#batch dim"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
    infinity: float = 1e8,
) -> Float[Tensor, "*batch dim-1"]:
    """project points in camera coordinate to image plane"""

    points = points / (points[..., -1:] + epsilon)
    points = points.nan_to_num(posinf=infinity, neginf=-infinity)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :-1]


def project(
    points: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Float[Tensor, "*batch dim-1"],  # xy coordinates
    Bool[Tensor, " *batch"],  # whether points are in front of the camera
]:
    """project point in world coordinate to image plane"""

    points = homogenize_points(points)
    points = transform_world2cam(points, extrinsics)[..., :-1]
    in_front_of_camera = points[..., -1] >= 0
    return project_camera_space(points, intrinsics, epsilon=epsilon), in_front_of_camera


def unproject(
    coordinates: Float[Tensor, "*#batch dim"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def normal2rotation(z):
    batch_size = z.shape[0]
    z = z / z.norm(dim=1, keepdim=True)

    # Generate a reference vector
    ref_vector = torch.tensor([1.0, 0.0, 0.0], device=z.device).repeat(batch_size, 1)
    parallel_mask = (torch.abs(z[:, 0]) > 0.99).unsqueeze(1)
    ref_vector[parallel_mask[:, 0]] = torch.tensor([0.0, 1.0, 0.0], device=z.device)

    # Project ref vector onto the plane orthogonal to z and normalize to get x-axis
    projections = (ref_vector * z).sum(dim=1, keepdim=True) * z
    x = ref_vector - projections
    x = x / x.norm(dim=1, keepdim=True)

    # Compute the y-axis as the cross product of z and x
    y = torch.cross(z, x, dim=1)
    y = y / y.norm(dim=1, keepdim=True)
    rotation = torch.stack([x, y, z], dim=-1)
    q = rotmat2quaternion(rotation)
    return q, rotation


def quaternion2rotmat(q):
    r, x, y, z = q.split(1, -1)
    # R = torch.eye(4).expand([len(q), 4, 4]).to(q.device)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - r * z),
            2 * (x * z + r * y),
            2 * (x * y + r * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - r * x),
            2 * (x * z - r * y),
            2 * (y * z + r * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )

    # Reshape to [batch_size, 3, 3]
    R = R.reshape(q.size(0), 3, 3)
    return R


def rotmat2quaternion(R, normalize=True):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    # print(torch.sum(torch.isnan(r)))
    q = torch.stack(
        [
            r,
            (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
            (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
            (R[:, 1, 0] - R[:, 0, 1]) / (4 * r),
        ],
        -1,
    )
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q


def get_world_rays(
    coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+2 dim+2"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> tuple[
    Float[Tensor, "*batch dim+1"],  # origins
    Float[Tensor, "*batch dim+1"],  # directions
]:
    """get rays in world coordinate"""

    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    # directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """

    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def voxel_downsample(point_cloud, voxel_size=0.02, num_points_per_voxel=1):
    # Step 1: Compute voxel indices for each point
    voxel_indices = torch.floor(point_cloud / voxel_size).long()

    # Step 2: Compute unique voxel indices and their corresponding counts
    unique_voxel_indices, inverse_indices = torch.unique(
        voxel_indices, return_inverse=True, dim=0
    )

    # Step 3: Shuffle the inverse indices
    rand_indices = torch.randperm(inverse_indices.size(0), device=point_cloud.device)
    shuffled_inverse_indices = inverse_indices[rand_indices]

    # Step 4: Select one index per unique voxel
    selected_indices = torch.zeros(
        unique_voxel_indices.shape[0], dtype=torch.long, device=point_cloud.device
    )
    selected_indices[shuffled_inverse_indices] = rand_indices

    # Step 5: Ensure only unique indices are selected
    selected_indices = selected_indices.unique()

    return selected_indices


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def render_cuda_core(
    cam_pos,
    fov: Float[Tensor, "2"],
    view_matrix: Float[Tensor, "4 4"],
    projection_matrix: Float[Tensor, "4 4"],
    render_mask,
    raydir_map,
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "3"],
    gaussian_means: Float[Tensor, "gaussian 3"],
    gaussian_sh_coefficients: Float[Tensor, "gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "gaussian"],
    gaussian_confidences: Float[Tensor, "gaussian"],
    gaussian_scales,
    gaussian_rotations,
    front_only=False,
    require_importance: bool = False,
    use_sh: bool = False,
    weight_thres=0.03,
) -> Float[Tensor, "batch 3 height width"]:

    front_config = 0.0
    if front_only:
        front_config = 1.0

    importance_config = 0.0
    if require_importance:
        importance_config = 1.0

    device = gaussian_means.device
    tan_fov = (0.5 * fov).tan()
    means_2d = torch.zeros_like(gaussian_means, requires_grad=True)
    try:
        means_2d.retain_grad()
    except Exception:
        pass

    settings = GaussianRasterizationSettings(
        image_height=image_shape[0],
        image_width=image_shape[1],
        tanfovx=tan_fov[0].item(),
        tanfovy=tan_fov[1].item(),
        bg=background_color,
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=projection_matrix,
        sh_degree=0,
        campos=cam_pos,
        prefiltered=False,  # This matches the original usage.
        render_mask=render_mask,
        weight_thres=weight_thres,
        debug=False,
        config=torch.tensor([1.0, 1.0, 1.0, importance_config, front_config]).to(
            device
        ),
    )
    rasterizer = GaussianRasterizer(settings)

    rgb, normal, depth, opacity, confidence, importance, count, radii = rasterizer(
        means3D=gaussian_means,
        means2D=means_2d,
        opacities=gaussian_opacities[..., None],
        confidences=gaussian_confidences,
        shs=None,
        colors_precomp=None if use_sh else gaussian_sh_coefficients[:, 0, :],
        scales=gaussian_scales,
        rotations=gaussian_rotations,
        cov3D_precomp=None,  # gaussian_covariances[:, row, col],
    )
    mask = opacity.detach() > 1e-2
    normal = torch.nn.functional.normalize(normal, dim=0) * mask
    visible_mask = torch.sum(normal * raydir_map, dim=0) < 0.0
    # confidence *= visible_mask.long()
    d2n = depth2normal(depth, mask, fov)

    return rgb, depth, normal, opacity, d2n, confidence, importance, count, radii


class GaussianRenderer:
    def __init__(
        self,
        extrinsics,
        intrinsics,
        gaussians_attr,
        background_color,
        near_far,
        resolution,
        device,
        render_masks=None,
    ):
        self.device = device
        (
            self.gaussian_means,
            self.gaussian_harmonics,
            self.gaussian_opacities,
            self.gaussian_confidences,
            self.gaussian_scales,
            self.gaussian_rotations,
        ) = gaussians_attr

        self.background_color = background_color
        self.h, self.w = resolution

        self.batch_size, _, _ = extrinsics.shape
        self.cam_pos = extrinsics[:, :3, 3]
        near = repeat(
            torch.tensor(near_far[0], device=self.device), "-> b", b=self.batch_size
        )
        far = repeat(
            torch.tensor(near_far[1], device=self.device), "-> b", b=self.batch_size
        )
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        self.fovs = torch.stack([fov_x, fov_y], dim=-1)
        projection_matrices_cam = get_projection_matrix(near, far, fov_x, fov_y)
        projection_matrices_cam = rearrange(projection_matrices_cam, "b i j -> b j i")
        self.view_matrices = rearrange(extrinsics.inverse(), "b i j -> b j i")

        self.projection_matrices = self.view_matrices @ projection_matrices_cam

        xy_ray, _ = sample_image_grid((self.h, self.w), self.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        directions = unproject(
            xy_ray,
            torch.ones_like(xy_ray[..., 0]),
            intrinsics,
        ).view(self.h, self.w, -1, 3)
        raydir_maps = torch.nn.functional.normalize(directions, dim=-1)
        self.raydir_map = raydir_maps.permute(2, 3, 0, 1)[0]  # 3 H W
        if render_masks is None:
            self.render_masks = [
                torch.tensor([], device=self.device) for _ in range(self.batch_size)
            ]
        else:
            self.render_masks = render_masks

    def update_attr(self, gaussians_attr):
        (
            self.gaussian_means,
            self.gaussian_harmonics,
            self.gaussian_opacities,
            self.gaussian_confidences,
            self.gaussian_scales,
            self.gaussian_rotations,
        ) = gaussians_attr

    # render image at a certain view
    def render_view(
        self, i=0, require_grad=False, require_importance=False, front_only=False
    ):
        with torch.set_grad_enabled(require_grad):
            (rgb, depth, normal, opacity, d2n, confidence, importance, count, radii) = (
                render_cuda_core(
                    self.cam_pos[i],
                    self.fovs[i],
                    self.view_matrices[i],
                    self.projection_matrices[i],
                    self.render_masks[i],
                    self.raydir_map,
                    (self.h, self.w),
                    self.background_color,
                    self.gaussian_means,
                    self.gaussian_harmonics,
                    self.gaussian_opacities,
                    self.gaussian_confidences,
                    self.gaussian_scales,
                    self.gaussian_rotations,
                    front_only=front_only,
                    require_importance=require_importance,
                )
            )
            in_frumstum_mask = radii > 0
        return (
            rgb,
            depth,
            normal,
            opacity,
            d2n,
            confidence,
            importance,
            count,
            in_frumstum_mask,
        )

    # render all images
    def render_view_all(
        self, require_grad=False, require_importance=False, front_only=False
    ):
        num_gaussians = len(self.gaussian_means)
        all_rgbs = torch.empty(self.batch_size, 3, self.h, self.w, device=self.device)
        all_depths = torch.empty(self.batch_size, 1, self.h, self.w, device=self.device)
        all_d2ns = torch.empty(self.batch_size, 3, self.h, self.w, device=self.device)
        all_normals = torch.empty(
            self.batch_size, 3, self.h, self.w, device=self.device
        )
        all_opacities = torch.empty(
            self.batch_size, 1, self.h, self.w, device=self.device
        )
        all_confidences = torch.empty(
            self.batch_size, 1, self.h, self.w, device=self.device
        )
        all_importances = torch.zeros(
            self.batch_size, num_gaussians, device=self.device
        )
        all_counts = torch.zeros(
            self.batch_size, num_gaussians, device=self.device, dtype=torch.int32
        )
        all_radiis = torch.zeros(num_gaussians, device=self.device, dtype=torch.int32)

        with torch.set_grad_enabled(require_grad):
            for i in range(self.batch_size):
                (
                    rgb,
                    depth,
                    normal,
                    opacity,
                    d2n,
                    confidence,
                    importance,
                    count,
                    radii,
                ) = render_cuda_core(
                    self.cam_pos[i],
                    self.fovs[i],
                    self.view_matrices[i],
                    self.projection_matrices[i],
                    self.render_masks[i],
                    self.raydir_map,
                    (self.h, self.w),
                    self.background_color,
                    self.gaussian_means,
                    self.gaussian_harmonics,
                    self.gaussian_opacities,
                    self.gaussian_confidences,
                    self.gaussian_scales,
                    self.gaussian_rotations,
                    front_only=front_only,
                    require_importance=require_importance,
                )

                all_rgbs[i] = rgb
                all_normals[i] = normal
                all_depths[i] = depth
                all_d2ns[i] = d2n
                all_opacities[i] = opacity
                all_confidences[i] = confidence
                all_importances[i] = importance
                all_counts[i] = count
                all_radiis += radii
        in_frumtum_mask = all_radiis > 0
        return (
            all_rgbs,
            all_depths,
            all_normals,
            all_opacities,
            all_d2ns,
            all_confidences,
            all_importances,
            all_counts,
            in_frumtum_mask,
        )
