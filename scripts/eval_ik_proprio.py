from libero.libero import benchmark
from diffusion_model.VideoGenerator import prepare_video_generator
from utils.env_utils import set_up_libero_envs
from utils.env_utils import process_obs
from policy.ik_model.resnet import ResNet50Pretrained, ResNet50
from model.resnet50_mlp import ResNet50MLP
from model.resnet18_mlp import ResNet18MLP
from model.cnn_mlp import CNNMLP

from model.vit_mlp import ViTMLP
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
from lightning.pytorch import seed_everything


# def correct_orientation(eight_d_output):
#     corrected_euler = st.Rotation.from_quat(eight_d_output[3:7]).as_euler("xzy", degrees=False)
#     corrected_euler[2] *= -1
#     corrected_quat = st.Rotation.from_euler("xzy", corrected_euler, degrees=False).as_quat()
    
#     new_8d = np.concatenate(
#         (
#             eight_d_output[:3].reshape(-1,),
#             corrected_quat.reshape(-1,),
#             eight_d_output[7].reshape(-1,),
#         )
#     )
#     return new_8d

# SPATIAL_TASK_DICT = {
#     "task1": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
#     "task2": "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
#     "task3": "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
#     "task4": "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
#     "task5": "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
#     "task6": "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
#     "task7": "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
#     "task8": "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
#     "task9": "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
#     "task10": "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
# }

# LB90_TASK_DICT = {

def main():
    # Load config
    with open("scripts/eval_ik.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Only set seed if use_seed is true
    if config.get('use_seed', False):
        seed_everything(config['seed'], workers=True)
        exp_name = f"seed-{config['seed']}_experiment"
    else:
        exp_name = "seed-None_experiment"
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"./results/ik_policy/{timestamp}_{exp_name}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device(f"cuda:{config['gpu_id']}")
    
    # Get tasks from LIBERO benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[config['task']['suite_name']]()
    available_tasks = suite.get_task_names()
    
    # Filter tasks if specific ones are selected in config
    if config['task']['selected_tasks']:
        selected_tasks = {f"task{i+1}": task_name 
                         for i, task_name in enumerate(available_tasks) 
                         if f"task{i+1}" in config['task']['selected_tasks']}
    else:
        # Use all available tasks if none specified
        selected_tasks = {f"task{i+1}": task_name 
                         for i, task_name in enumerate(available_tasks)}
    
    transform = transforms.Compose(
        [
            transforms.ToPILImage(), 
            transforms.Resize(config['image']['resize']), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    video_generator = prepare_video_generator(
        unet_path=config['video_generator_path'], 
        device=device, 
        sample_timestep=config['video']['sample_timestep']
    )
    for param in video_generator.parameters():
        param.requires_grad = False
    # ik_model = ResNet50MLP(out_size=8).to(device)
    # Initialize IK model based on config
    if config['model']['type'] == 'vit':
        ik_model = ViTMLP(
            out_size=config['model']['out_size'], 
            pretrained=config['model']['pretrained']
            
        ).to(device)
    elif config['model']['type'] == 'resnet50':
        ik_model = ResNet50MLP(
            out_size=config['model']['out_size']
        ).to(device)
    elif config['model']['type'] == 'cnn':
        ik_model = CNNMLP(
            out_size=config['model']['out_size']
        ).to(device)
    elif config['model']['type'] == 'resnet18':
        ik_model = ResNet18MLP(
            out_size=config['model']['out_size']
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
        
    ik_model.load_state_dict(torch.load(config['ik_model_path']))
    ik_model.eval()

    all_task_results = {}
    
    for task_id, task_name in selected_tasks.items():
        print(f"\nEvaluating {task_id}: {task_name}")
        task_output_dir = os.path.join(base_output_dir, task_id)
        os.makedirs(task_output_dir, exist_ok=True)
        
        env, task_prompt = set_up_libero_envs(
            suite_name=config['task']['suite_name'], 
            task_name=task_name, 
            render_device=config['render_gpu_id'],
            horizon=config['task']['horizon']
        )
        task_successes = 0

        for test_time in tqdm(range(config['num_test_pr_task'])):
            obs = env.reset()
            for _ in range(config['init_steps']):
                obs, _, done, _ = env.step(np.zeros(7))

            roll_out_video = []
            for _ in range(config['num_video_samples']):
                visual_obs, _ = process_obs(obs, extra_state_keys=[], device=device)
                side_view = visual_obs[:,0]
                # print("##########################")
                # print(side_view.shape)
                # # print side view type
                # print(type(side_view))
                # print(side_view.dtype)
                video_clip = video_generator(task_prompt, side_view)
                video_clip = rearrange(video_clip, "b (f c) h w -> (b f) c h w", c=3) # [::2]
                video_clip = (video_clip.permute(0, 2, 3, 1).detach().cpu().numpy()*255).astype(np.uint8)
                
                vhrz = config['video']['act_horizon']
                assert len(video_clip) > vhrz, "Video clip length must be greater than act_horizon"
                
                for index in range(vhrz):
                    goal_img = video_clip[index]
                    goal_img = transform(goal_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        goal_out = ik_model(goal_img)
                    goal_out = goal_out.squeeze().cpu().numpy()
                    # goal_out = correct_orientation(goal_out)
                    
                    # Predicted proprioceptive state
                    img_st = obs["agentview_image"]
                    img_st = cv2.flip(img_st, 0)
                    img_st = transform(img_st).unsqueeze(0).to(device)

                    # with torch.no_grad():
                    #     st_out = ik_model(img_st)
                    # st_out = st_out.squeeze().cpu().numpy()
                    # st_out = correct_orientation(st_out)
                    
                    # Get proprioceptive state
                    # proprio_st = np.concatenate([obs["robot0_eef_pos"].reshape(3,), obs["robot0_eef_quat"].reshape(4,), obs["robot0_gripper_qpos"][:, 0].reshape(1,)], axis=0)
                    # self.true_now = np.concatenate([self.obs["robot0_eef_pos"].reshape(3,), self.obs["robot0_eef_quat"].reshape(4,), self.obs["robot0_gripper_qpos"][:, 0].reshape(1,)], axis=0)
                    try:
                        gripper_pos = obs["robot0_gripper_qpos"]
                        if isinstance(gripper_pos, np.ndarray):
                            gripper_value = gripper_pos.flatten()[0:1]
                        else:
                            gripper_value = np.array([float(gripper_pos)]).reshape(1,)
                            
                        proprio_st = np.concatenate([
                            obs["robot0_eef_pos"].reshape(3,),
                            obs["robot0_eef_quat"].reshape(4,),
                            gripper_value
                        ], axis=0)
                    except Exception as e:
                        print("Error processing gripper position:", e)
                        print("Gripper position data:", obs["robot0_gripper_qpos"])
                        raise
                    

                    pid_step = 0
                    prev_error = np.zeros(7)
                    integral_error = np.zeros(7)
                            
                    while np.linalg.norm(proprio_st - goal_out) > config['pid']['convergence_threshold'] and pid_step < config['pid_max_steps']:
                        pid_step += 1

                        time.sleep(0.01)
                        
                        control_trans = goal_out[:3] - proprio_st[:3]
                        control_quat = (st.Rotation.from_quat(goal_out[3:7])* st.Rotation.from_quat(proprio_st[3:7]).inv())
                        control_rot = control_quat.as_euler("xyz", degrees=False)

                        gripper_goal = goal_out[7] if goal_out[7] > config['pid']['grip_threshold'] else 0.00
                        control_gripper = (gripper_goal - proprio_st[7]) * 0.04

                        ik_act = np.zeros(7)
                        ik_act[:3] = config['pid']['pos_scale'] * control_trans
                        ik_act[3:6] = config['pid']['rot_scale'] * control_rot
                        ik_act[6] = config['pid']['grip_scale'] * control_gripper
                        ik_act = ik_act.astype(np.float32)
                        
                        prev_error = ik_act
                        integral_error += ik_act
                        control_signal = (config['pid']['kp'] * ik_act + 
                                        config['pid']['ki'] * integral_error + 
                                        config['pid']['kd'] * (ik_act - prev_error))
                        
                        obs, _, done, _ = env.step(control_signal)
                        
                        img_st = obs["agentview_image"]
                        img_st = cv2.flip(img_st, 0)
                        img_st = transform(img_st).unsqueeze(0).to(device)
                        # with torch.no_grad():
                        #     st_out = ik_model(img_st)
                        # st_out = st_out.squeeze().cpu().numpy()
                        # # st_out = correct_orientation(st_out)
                        try:
                            gripper_pos = obs["robot0_gripper_qpos"]
                            if isinstance(gripper_pos, np.ndarray):
                                gripper_value = gripper_pos.flatten()[0:1]
                            else:
                                gripper_value = np.array([float(gripper_pos)]).reshape(1,)
                                
                            proprio_st = np.concatenate([
                                obs["robot0_eef_pos"].reshape(3,),
                                obs["robot0_eef_quat"].reshape(4,),
                                gripper_value
                            ], axis=0)
                        except Exception as e:
                            print("Error processing gripper position:", e)
                            print("Gripper position data:", obs["robot0_gripper_qpos"])
                            raise
                        
                        
                        frame = np.concatenate([cv2.flip(obs["agentview_image"], 0), video_clip[index]], axis=1)
                        cv2.putText(frame, "Current", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, "Goal", (frame.shape[1]//2 + 10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        roll_out_video.append(frame)
                        
                        if done:
                            break

                    if done:
                        break

                    time.sleep(0.01)
            if done:
                print(f"Test {test_time}: Success!")
                status = "success"
                task_successes += 1
            else:
                print(f"Test {test_time}: Fail!")
                status = "fail"

            time.sleep(0.01)
                
            # Save as GIF instead of MP4
            video_name = f"test_{test_time:02d}_{status}_steps{len(roll_out_video)}.gif"
            video_path = os.path.join(task_output_dir, video_name)
            
            with imageio.get_writer(video_path, mode='I', duration=config['video']['duration']) as writer:
                for frame in roll_out_video:
                    writer.append_data(frame)
            
            # Log individual test result
            info_path = os.path.join(task_output_dir, "task_info.txt")
            with open(info_path, "a") as f:
                f.write(f"Test {test_time}: {'Success' if done else 'Fail'}\n")

        env.close()
        
        # Calculate and save task-specific results
        task_success_rate = (task_successes / config['num_test_pr_task']) * 100
        all_task_results[task_id] = task_success_rate
        
        with open(os.path.join(task_output_dir, "task_results.txt"), "w") as f:
            f.write(f"Task: {task_name}\n")
            f.write(f"Success rate: {task_success_rate:.2f}%\n")
            f.write(f"Successful attempts: {task_successes}/{config['num_test_pr_task']}\n")
            f.write(f"PID params: kp={config['pid']['kp']}, ki={config['pid']['ki']}, kd={config['pid']['kd']}\n")
    
    # Save overall results
    overall_success_rate = sum(all_task_results.values()) / len(all_task_results)
    with open(os.path.join(base_output_dir, "overall_results.txt"), "w") as f:
        f.write(f"Experiment Timestamp: {timestamp}\n")
        if config.get('use_seed', False):
            f.write(f"Random Seed: {config['seed']}\n")
        f.write(f"Overall success rate across all tasks: {overall_success_rate:.2f}%\n\n")
        f.write("Individual Task Results:\n")
        for task_id, success_rate in all_task_results.items():
            f.write(f"{task_id}: {success_rate:.2f}%\n")
        f.write(f"\nPID params: kp={config['pid']['kp']}, ki={config['pid']['ki']}, kd={config['pid']['kd']}\n")

main()