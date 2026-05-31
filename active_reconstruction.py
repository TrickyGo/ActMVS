import torch
import warnings
import os
import yaml

from tools.logger import Logger
from mapping.mapper import Mapper
from planning.planner import Planner
from simulator.habitat_simulator import Simulator

warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cfg = load_yaml_config('config.yaml')
    experiment_path = os.path.join(
        cfg["experiment"]["output_dir"],
        cfg["experiment"]["scene_name"],
    )
    os.makedirs(experiment_path, exist_ok=True)
    cfg["vis_dir"] = os.path.join(
        experiment_path,
        'vis'
    )
    os.makedirs(cfg["vis_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["vis_dir"],'render'), exist_ok=True)
    logger = Logger(experiment_path, cfg)

    simulator = Simulator(cfg)
    mapper = Mapper(cfg, device)
    planner = Planner(cfg, device)
    mapper.load_recorder(logger)
    mapper.load_simulator(simulator)
    mapper.load_planner(planner)

    mapper.run()

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

if __name__ == "__main__":
    main()

