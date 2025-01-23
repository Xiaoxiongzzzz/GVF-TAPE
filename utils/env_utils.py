from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import os
import cv2
import numpy as np
import torch
import torchvision

states_key_mapping = {
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
}


def set_up_libero_envs(suite_name: str, task_name: str, render_device: int, horizon: int):
    '''
    Args:
    suite_name: e.g. libero_spatial, libero_goal ...
    task_name: e.g. "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"

    Return:
    env: OffScreenRenderEnv
    task_prompt: str e.g. "pick up the black bowl from table center and place it on the plate"
    '''
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()

    task_id = task_suite.get_task_names().index(task_name)
    task = task_suite.get_task(task_id)
    task_prompt = task.language

    bddl_files_path = '/mnt/home/ZhangXiaoxiong/Documents/VideoGeneration/third_party/LIBERO/libero/libero/bddl_files'
    task_bddl_file = os.path.join(bddl_files_path, task.problem_folder, task.bddl_file)

    env_args ={
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
    "render_gpu_device_id": render_device, 
    "has_renderer": True,
    "horizon": horizon,
    "initialization_noise": {"magnitude": 0.00000, "type": "gaussian"},
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()

    initial_states = task_suite.get_task_init_states(task_id)
    init_state_id = 0
    env.set_init_state(initial_states[init_state_id])

    return env, task_prompt

def process_obs(obs, extra_state_keys, device):
    ''' 
    get the needed input in specified dtype and load to device
    Args:
    obs: the raw observation output by the environment
    extra_state_keys: e.g. ["joint_states", "gripper_states"]
    device: torch.device
    Retrun:
    visual_obs: [b, view, channel, height, width]
    extra_states: {k: [b, dim]}
    NOTE: NOT Nomarlized (0-255)
    '''
    agent_view = torch.flip(torch.from_numpy(obs['agentview_image']).to(device), dims=(0,))            #[height, width, channel]
    eye_in_hand = torch.flip(torch.from_numpy(obs['robot0_eye_in_hand_image']).to(device), dims=(0,))  #[height, width, channel]
    visual_obs = torch.stack([agent_view, eye_in_hand], dim=0).unsqueeze(0)      # [batch, view, height, width, channel]
    visual_obs = visual_obs.permute(0, 1, 4, 2, 3)      #[batch, view, channel, height, width]

    extra_states = {k: obs[states_key_mapping[k]] for k in extra_state_keys}       #{k: [dims,]}
    extra_states = {k: torch.from_numpy(v).unsqueeze(0) for k, v in extra_states.items()}   #{k: [batch, dims,]}
    extra_states = {k: v.to(torch.float32).to(device) for k, v in extra_states.items()}

    return visual_obs, extra_states

