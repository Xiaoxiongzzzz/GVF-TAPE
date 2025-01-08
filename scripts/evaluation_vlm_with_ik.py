from libero.libero import benchmark
from diffusion_model.VideoGenerator import prepare_video_generator
from utils.env_utils import set_up_libero_envs
from utils.env_utils import process_obs
from policy.ik_model.resnet import ResNet50Pretrained, ResNet50
from torchvision.utils import save_image
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import numpy as np
import os
import cv2
import torch
import time
import yaml
import imageio

mean = np.array([[-0.02596965, 0.10603805, 1.05615763, 0.96777721, -0.04932071, -0.03729568, 0.04566995, 0.01967621]])
std = np.array([[0.11212396, 0.1382502, 0.10310984, 0.05012393, 0.19773889, 0.11159955, 0.05857035, 0.0170986]])

def main():
    device = torch.device("cuda:1")
    suite_name = "libero_spatial"
    task_name = "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
    control_freq = 15

    video_generator = prepare_video_generator(unet_path="/mnt/home/ZhangXiaoxiong/Documents/AVDC/results/RFlow_pixel/model_99000.pt", device=device, sample_timestep=3)
    for param in video_generator.parameters():
        param.requires_grad = False
    ik_model = ResNet50Pretrained(output_dim=8).to(device)
    ik_model.load_state_dict(torch.load("/mnt/home/ZhangXiaoxiong/Documents/train_libero_spatial_ik_ResNet50MLP_final.pth"))
    ik_model.eval()

    env, task_prompt = set_up_libero_envs(suite_name, task_name, render_device=0)

    for test_time in tqdm(range(10)):
        obs = env.reset()
        for _ in range(5):
            obs, _, done, _ = env.step(np.zeros(7))

        roll_out_video = []
        for _ in range(13):
            visual_obs, _ = process_obs(obs, extra_state_keys=[], device=device)
            side_view = visual_obs[:,0]
            video_clip = video_generator(task_prompt, side_view)
            video_clip = rearrange(video_clip, "b (f c) h w -> (b f) c h w", c=3)[::2]
  
            pred_pose = ik_model(video_clip).detach().cpu().numpy()
            video_clip = (video_clip.permute(0, 2, 3, 1).detach().cpu().numpy()*255).astype(np.uint8)
            for index, goal_pose in enumerate(pred_pose):
                for control_time in range(control_freq):
                    translation = goal_pose[:3] - obs["robot0_eef_pos"]
                    orientation_current = R.from_quat(obs["robot0_eef_quat"])
                    orientation_goal = R.from_quat(goal_pose[-5:-1])
                    rotation = rotation_control(orientation_goal, orientation_current)

                    if goal_pose[-1] > 0.7:
                        gripper_control = (obs["robot0_gripper_qpos"][0]) - goal_pose[-1]
                    else:
                        gripper_control = 1

                    control = np.concatenate([translation*control_time*0.75, rotation*0.1, [gripper_control]], axis=0)
                    obs, _, done, _ = env.step(control)

                    frame = np.concatenate([cv2.flip(obs["agentview_image"], 0), video_clip[index]], axis=1)
                    cv2.putText(frame, f"gripper est. {goal_pose[-1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    roll_out_video.append(frame)
                    
                    if done:
                        break
                if done:
                    break
        if done:
            print("Success!")
        else:
            print("Fail!")
        with imageio.get_writer(f"./results/ik_policy/video_{test_time}.gif", mode='I', duration=50) as writer:
            for image in roll_out_video:
                writer.append_data(image)
    env.close()
def rotation_control(orientation_goal, orientation_current):
    """Get SO(3) input and return 'xyz' delta Euler angle"""
    orientation_goal = orientation_goal.as_euler('xzy', degrees=False)
    orientation_goal[2] = -orientation_goal[2]
    orientation_goal = R.from_euler('xzy', orientation_goal, degrees=False)
    rotation_operation = orientation_goal * orientation_current.inv()
    rotation_operation = rotation_operation.as_euler('xyz', degrees=False)
    return rotation_operation

def exponential_smoothing(sequence, alpha):
    """Exponential smoothing for a sequence of values (Multi Dimentional)
        Args:
            alpha: is the weight for current data
    """
    smoothed_sequence = np.zeros_like(sequence)
    smoothed_sequence[0] = sequence[0]

    for i in range(1, len(sequence)):
        smoothed_sequence[i] = alpha * sequence[i] + (1 - alpha) * smoothed_sequence[i - 1]

    return smoothed_sequence

main()