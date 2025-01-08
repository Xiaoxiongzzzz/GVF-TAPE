from utils.env_utils import set_up_libero_envs
from libero.libero import benchmark
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import cv2
import h5py
import imageio
benchmark_dict = benchmark.get_benchmark_dict()
suite_name = "libero_spatial"
suite = benchmark_dict[suite_name]()
task_names = suite.get_task_names()
task_name = task_names[0]
env, task_prompt = set_up_libero_envs(suite_name, task_name, 1)
obs = env.reset()
with h5py.File("/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5") as f:
    demo = f["data"]["demo_0"]
    images = demo["obs"]["agentview_rgb"][:]
    ee_states = demo["obs"]["ee_states"][:]
    griper_states = demo["obs"]["gripper_states"][:, 0] - demo["obs"]["gripper_states"][:, 1]
    video_clip = []
    done = False
    for index, goal_state in enumerate(ee_states):
        for _ in range(5):
            pos_control = goal_state[:3] - obs["robot0_eef_pos"]

            orientation_current = R.from_quat(obs["robot0_eef_quat"])
            orientation_goal = R.from_euler('xzy', goal_state[-3:], degrees=False)
            rotation_operation = orientation_goal * orientation_current.inv()
            rotation_control = rotation_operation.as_euler('xyz', degrees=False)

            gripper_control = (obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]) -  0.04
            control = np.concatenate([pos_control*10, rotation_control*10, [gripper_control]], axis=0)
            obs, reward, done, _ = env.step(control)
            video_clip.append(np.concatenate([cv2.flip(obs["agentview_image"], 0), cv2.flip(images[index], 0)], axis=1))
            if done:
                break
        if done:
            break
    env.close()
# os.makedirs("./results/ik_policy", exist_ok=True)
imageio.mimsave("./demo.gif", video_clip, duration=100)