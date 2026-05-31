import habitat_sim
import numpy as np
import quaternion
import torch
import os
import logging
import trimesh
from .utils import *

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


class Simulator:
    def __init__(self, cfg):
        experiment_cfg = cfg["experiment"]
        print("\n ----------configure simulator----------")

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.gpu_device_id = 0
        assert os.path.exists(experiment_cfg["scene_id"])
        backend_cfg.scene_id = experiment_cfg["scene_id"]
        backend_cfg.enable_physics = False
        self.scene_name = experiment_cfg["scene_name"]
        self.has_missing_surface = experiment_cfg["has_missing_surface"]
        self.mesh = trimesh.load(experiment_cfg["mesh_path"])
        self.bbox = np.array(self.mesh.bounding_box.bounds)

        sensor_specs = []
        self.resolution = np.array([512, 512])
        H, W = self.resolution
        self.fov = np.array([90, 90])
        vfov, hfov = self.fov
        self.intrinsic = compute_camera_intrinsic(
            H, W, vfov, hfov, normalize=True
        )

        for sensor_type in ["color"]:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_type
            sensor_spec.sensor_type = SENSOR_TYPE[sensor_type]
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_spec.resolution = [H, W]
            sensor_spec.vfov = vfov
            sensor_spec.hfov = hfov
            sensor_spec.position = [0.0, 0.0, 0.0]
            sensor_specs.append(sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        print(
            "\n Simulator Loading Scene:",
            self.scene_name,
        )

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)

        print("\n ----------load habitat simulator----------")
        self.data = {}

    def simulate(self, c2w):
        c2w_habitat = opencv_to_opengl_camera(c2w.numpy())
        orientation = quaternion.from_rotation_matrix(c2w_habitat[:3, :3])
        position = np.array(c2w_habitat[:3, 3])
        agent_state = habitat_sim.agent.AgentState(position, orientation)
        self.sim.get_agent(0).set_state(agent_state)

        obs = self.sim.get_sensor_observations()
        color = obs.get("color", None)
        if color is not None:
            rgb = color[:, :, :3] / 255.0
            rgb = torch.from_numpy(rgb.astype(np.float32))
            rgb = rgb.permute(2, 0, 1)

        data_frame = {
            "extrinsic": c2w,
            "intrinsic": self.intrinsic,
            "rgb": rgb,
        }

        return data_frame
