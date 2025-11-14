import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

from hydra.core.global_hydra import GlobalHydra

# 清除已存在的 GlobalHydra 实例
GlobalHydra.instance().clear()

SPLIT = "mini"  # ["mini", "test", "trainval"]
FILTER = "all_scenes"

hydra.initialize(config_path="./navsim/planning/script/config/common/train_test_split/scene_filter")
cfg = hydra.compose(config_name=FILTER)

scene_filter: SceneFilter = instantiate(cfg)
openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig

agent = TransfuserAgent(config=TransfuserConfig(), lr=1e-4)


scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}", # data_path
    openscene_data_root / f"sensor_blobs/{SPLIT}", # original_sensor_path
    scene_filter,
    openscene_data_root / "warmup_two_stage/sensor_blobs", # synthetic_sensor_path
    openscene_data_root / "warmup_two_stage/synthetic_scene_pickles", # synthetic_scenes_path
    sensor_config=agent.get_sensor_config(),
)

token = np.random.choice(scene_loader.tokens)

scene = scene_loader.get_scene_from_token('ca9e7281adce5212')

# print(scene)

# from navsim.visualization.plots import plot_bev_frame
# frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame
# fig, ax = plot_bev_frame(scene, frame_idx)


from navsim.visualization.plots import plot_bev_with_agent
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.agents.ego_status_mlp_agent import EgoStatusMLPAgent
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.transfuser.transfuser_config import TransfuserConfig

# 未来有10帧，过去有4帧，改time_horizon，
agent = ConstantVelocityAgent(trajectory_sampling=TrajectorySampling(time_horizon=2, interval_length=0.5))
agent = EgoStatusMLPAgent(8, 0.01)
agent = TransfuserAgent(config=TransfuserConfig(), lr=1e-4)
fig, ax = plot_bev_with_agent(scene, agent)
