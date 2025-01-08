from libero.libero import benchmark
from diffusion_model.VideoGenerator import prepare_video_generator
from utils.env_utils import set_up_libero_envs
from utils.env_utils import process_obs
from policy.ik_model.resnet import ResNet50Pretrained, ResNet50
from model.resnet50_mlp import ResNet50MLP
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
import torchvision.transforms as transforms
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R



# mean = np.array([[-0.02596965, 0.10603805, 1.05615763, 0.96777721, -0.04932071, -0.03729568, 0.04566995, 0.01967621]])
# std = np.array([[0.11212396, 0.1382502, 0.10310984, 0.05012393, 0.19773889, 0.11159955, 0.05857035, 0.0170986]])

transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

def correct_orientation(eight_d_output):
    """
    orientation correction to fit data format
    """
    
    corrected_euler = st.Rotation.from_quat(eight_d_output[3:7]).as_euler("xzy", degrees=False)
    corrected_euler[2] *= -1
    corrected_quat = st.Rotation.from_euler("xzy", corrected_euler, degrees=False).as_quat()
    
    new_8d = np.concatenate(
        (
            eight_d_output[:3].reshape(-1,),
            corrected_quat.reshape(-1,),
            eight_d_output[7].reshape(-1,),
        )
    )
    
    return new_8d

def main():
    # Add timestamp for unique experiment folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/ik_policy/experiment_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    device = torch.device("cuda:1")
    suite_name = "libero_spatial"
    task_name = "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
    control_freq = 15

    video_generator = prepare_video_generator(unet_path="/mnt/home/ZhangXiaoxiong/Documents/AVDC/results/RFlow_pixel/model_99000.pt", device=device, sample_timestep=3)
    for param in video_generator.parameters():
        param.requires_grad = False
    ik_model = ResNet50MLP(out_size=8).to(device)
    ik_model.load_state_dict(torch.load("/mnt/home/ZhangXiaoxiong/Documents/train_libero_spatial_ik_ResNet50MLP_final.pth"))
    ik_model.eval()

    env, task_prompt = set_up_libero_envs(suite_name, task_name, render_device=0)

    for test_time in tqdm(range(10)):
        # Create subfolder for this test
        # test_dir = os.path.join(output_dir, f"test_{test_time}")
        # os.makedirs(test_dir, exist_ok=True)

        obs = env.reset()
        for _ in range(5):
            obs, _, done, _ = env.step(np.zeros(7))

        roll_out_video = []
        for _ in range(13):
            visual_obs, _ = process_obs(obs, extra_state_keys=[], device=device)
            side_view = visual_obs[:,0]
            video_clip = video_generator(task_prompt, side_view)
            video_clip = rearrange(video_clip, "b (f c) h w -> (b f) c h w", c=3)[::2]
  
            # pred_pose = ik_model(video_clip).detach().cpu().numpy()
            video_clip = (video_clip.permute(0, 2, 3, 1).detach().cpu().numpy()*255).astype(np.uint8)
            
            
            for index in range(len(video_clip)):
                
                goal_img = video_clip[index]
                # transform goal_img
                goal_img = transform(goal_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    goal_out = ik_model(goal_img)
                goal_out = goal_out.squeeze().cpu().numpy()
                goal_out = correct_orientation(goal_out)
                
                img_st = obs["agentview_image"]
                # print(f"Warning: image shape: {img_st.shape}")
                # turn img_st upside down
                img_st = cv2.flip(img_st, 0)
                
                img_st = transform(img_st).unsqueeze(0).to(device)
                # print(f"Warning: image shape: {img_st.shape}")

                with torch.no_grad():
                    st_out = ik_model(img_st)
                st_out = st_out.squeeze().cpu().numpy()
                st_out = correct_orientation(st_out)
                
                # print st_out and goal_out
                # print(f"st_out: {st_out}")
                # print(f"goal_out: {goal_out}")
                
                # directly save two corresponding images for visualization
                # save img_st and goal_img
                # img_st = (img_st.permute(0, 2, 3, 1).detach().cpu().numpy()*255).astype(np.uint8)
                # goal_img = (goal_img.permute(0, 2, 3, 1).detach().cpu().numpy()*255).astype(np.uint8)
                # cv2.imwrite(os.path.join(output_dir, f"img_st_{index}.png"), img_st[0])
                # cv2.imwrite(os.path.join(output_dir, f"goal_img_{index}.png"), goal_img[0])
                
                pid_step = 0
                prev_error = np.zeros(7)
                integral_error = np.zeros(7)
                
                k_p = 0.2
                k_i = 0.01
                k_d = 0.05
                        
                while np.linalg.norm(st_out - goal_out) > 0.001 and pid_step < 16:
                    pid_step += 1
                    # delta control signal (action)
                    control_trans = goal_out[:3] - st_out[:3]
                    control_quat = (st.Rotation.from_quat(goal_out[3:7])* st.Rotation.from_quat(st_out[3:7]).inv())
                    control_rot = control_quat.as_euler("xyz", degrees=False)

                    # binary gripper function
                    gripper_goal = goal_out[7] if goal_out[7] > 0.7 else 0.00
                    control_gripper = (gripper_goal - st_out[7]) * 0.04

                    # put deltas together
                    ik_act = np.zeros(7)
                    ik_act[:3] = 100 * control_trans
                    ik_act[3:6] = control_rot
                    ik_act[6] = -10 * control_gripper

                    ik_act = ik_act.astype(np.float32)
                    
                    prev_error = ik_act
                    integral_error += ik_act
                    control_signal = (k_p * ik_act + k_i * integral_error + k_d * (ik_act - prev_error))
                    
                    # clip control signal
                    # control_signal = np.clip(control_signal, -0.5, 0.5)
                    
                    obs, _, done, _ = env.step(control_signal)
                    # log action carried
                    # print(f"action carried: {control_signal}")
                    
                    img_st = obs["agentview_image"]
                    img_st = cv2.flip(img_st, 0)
                    img_st = transform(img_st).unsqueeze(0).to(device)
                    with torch.no_grad():
                        st_out = ik_model(img_st)
                    st_out = st_out.squeeze().cpu().numpy()
                    st_out = correct_orientation(st_out)
                    
                    frame = np.concatenate([cv2.flip(obs["agentview_image"], 0), video_clip[index]], axis=1)
                    cv2.putText(frame, f"gripper est. {goal_out[-1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    roll_out_video.append(frame)
                    
                    if done:
                        break

                if done:
                    break
        if done:
            print(f"Test {test_time}: Success!")
            status = "success"
        else:
            print(f"Test {test_time}: Fail!")
            status = "fail"
            
        # Create informative filename
        video_name = f"test_{test_time:02d}_{status}_steps{len(roll_out_video)}.gif"
        video_path = os.path.join(output_dir, video_name)
        
        with imageio.get_writer(video_path, mode='I', duration=50) as writer:
            for image in roll_out_video:
                writer.append_data(image)
                
        # Save experiment info
        info_path = os.path.join(output_dir, "experiment_info.txt")
        with open(info_path, "a") as f:
            f.write(f"Test {test_time}: {'Success' if done else 'Fail'}\n")
            
    env.close()
    
    # Save overall results
    success_rate = len([f for f in os.listdir(output_dir) if "success" in f]) / 10 * 100
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"Overall success rate: {success_rate}%\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        # f.write(f"Model path: {video_generator.unet_path}\n")
        # f.write(f"IK model path: {ik_model_path}\n")
        f.write(f"PID params: kp={k_p}, ki={k_i}, kd={k_d}\n")

main()