import warnings

# Filter out specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*declare_namespace.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*lightning.fabric.*")
warnings.filterwarnings("ignore", message=".*lightning.pytorch.*")
warnings.filterwarnings("ignore", message=".*mpl_toolkits.*")
warnings.filterwarnings("ignore", message=".*google.*")
# Filter libero dataset path warnings
warnings.filterwarnings("ignore", message=".*datasets path.*does not exist.*")
# Filter additional warnings
warnings.filterwarnings("ignore", message=".*timm.layers.*")
warnings.filterwarnings("ignore", message=".*lightning_utilities.*")
warnings.filterwarnings("ignore", message=".*is deprecated as an API.*")
# Filter more specific warnings
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers is deprecated.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*implicit namespace packages.*")
# Filter namespace warnings
warnings.filterwarnings("ignore", message=".*declare_namespace('mpl_toolkits').*")
warnings.filterwarnings("ignore", message=".*declare_namespace('google').*")
warnings.filterwarnings("ignore", message=".*declare_namespace('lightning.fabric').*")
warnings.filterwarnings("ignore", message=".*declare_namespace('lightning').*")
warnings.filterwarnings("ignore", message=".*declare_namespace('lightning.pytorch').*")
# Filter huggingface warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*")
# Filter by module
warnings.filterwarnings("ignore", module="pkg_resources.*")
warnings.filterwarnings("ignore", module="lightning.*")

from libero.libero import benchmark
from diffusion_model.VideoGenerator import prepare_video_generator
from utils.env_utils import set_up_libero_envs, process_obs
from policy.ik_model.resnet import ResNet50Pretrained, ResNet50
from model.resnet50_mlp import ResNet50MLP
from model.resnet18_mlp import ResNet18MLP
from model.cnn_mlp import CNNMLP
from model.vit_mlp import ViTMLP
from model.depth_vit_mlp import DepthViTMLP
from model.cross_depth_vit_mlp import CrossDepthViTMLP
from model.all_cross_depth_vit_mlp import AllCrossDepthViTMLP
from model.cls_depth_vit_mlp import CLSDepthViTMLP
from model.depth_cross_rgb_vit_mlp import DepthCrossRGBViTMLP
from torchvision.utils import save_image
from einops import rearrange
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
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from collections import defaultdict
import time as timer
import multiprocessing as mp
import shutil
import torchvision

# torch.cuda.empty_cache()

# CONFIG_PATH = "conf/eval_rgb_ik.yaml"
# CONFIG_PATH = "conf/eval_play_cross_depth_encoder.yaml"
# CONFIG_PATH = "conf/eval_depth_cross_rgb.yaml"
# CONFIG_PATH = "conf/eval_play_depth_cross_rgb.yaml"
CONFIG_PATH = "conf/eval_rgb_expert_ik.yaml"


class IKEvaluator:
    def __init__(self, config_path=CONFIG_PATH):
        # Store config path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set basic parameters
        # self.device = torch.device(f"cuda:{self.config['gpu_id']}")
        self.device = torch.device(f"cuda:0")
        print(f"Using Computing GPU: {self.device}")
        
        torch.multiprocessing.set_sharing_strategy('file_system')
        
        self.num_processes = self.config['num_processes']
        if self.config['video']['depth']:
            self.channel_num = 4
        else:
            self.channel_num = 3
        
        # Set up experiment environment
        # torch.cuda.empty_cache()
        self._setup_experiment()
        
        # Load models
        self._setup_models()
        
        self.depth_transform = None
        
        self.transform = transforms.Compose([
                transforms.Resize(
                    (self.config['image']['resize'], self.config['image']['resize']), 
                    antialias=True,
                ),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        
        if self.config['video']['depth']:
            self.depth_transform = transforms.Compose([
                transforms.Resize(
                    (self.config['image']['resize'], self.config['image']['resize']), 
                    antialias=True,
                ),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            
        # Get task list
        self._setup_tasks()
        
    def _setup_experiment(self):
        """Set up experiment environment and output directories"""
        # Set random seed
        if self.config.get('use_seed', False):
            seed_everything(self.config['seed'], workers=True)
            seed_suffix = f"seed-{self.config['seed']}"
        else:
            seed_suffix = "seed-None"
            
        # Get output directory configuration
        base_dir = self.config.get('output', {}).get('base_dir', 
            "/mnt/data0/xiaoxiong/single_view_goal_diffusion/libero_results")
        exp_name = self.config.get('output', {}).get('exp_name', "default_experiment")
        
        # Get current timestamp
        timestamp = time.strftime("%Y%m%d_%H%M")
        
        # Create output directory with timestamp and seed
        self.base_output_dir = os.path.join(
            base_dir,
            f"{exp_name}_{timestamp}_{seed_suffix}"
        )
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # Copy config file to output directory
        config_filename = os.path.basename(self.config_path)
        shutil.copy2(self.config_path, os.path.join(self.base_output_dir, config_filename))
        
    def _setup_models(self):
        """Initialize video generator and IK model"""
        # Load video generator
        self.video_generator = prepare_video_generator(
            unet_path=self.config['video_generator_path'], 
            device=self.device, 
            sample_timestep=self.config['video']['sample_timestep'],
            latent=self.config['video']['latent'],
            depth=self.config['video']['depth'],
        )
        for param in self.video_generator.parameters():
            param.requires_grad = False
        
        # Load IK model
        model_type = self.config['model']['type']
        out_size = self.config['model']['out_size']
        
        if model_type == 'vit':
            self.ik_model = ViTMLP(
                out_size=out_size, 
                pretrained=self.config['model'].get('pretrained', True),
                img_height=self.config['image']['resize'],
                img_width=self.config['image']['resize'],
                model_name=self.config['model']['encoder_type'],
                in_channel=self.config['model']['in_channel']
            ).to(self.device)
        elif model_type == "cls_vit":
            self.ik_model = CLSDepthViTMLP(
                out_size=out_size, 
                pretrained=self.config['model'].get('pretrained', True),
                img_height=self.config['image']['resize'],
                img_width=self.config['image']['resize'],
                model_name=self.config['model']['encoder_type']
            ).to(self.device)
        elif model_type == 'depth_x_rgb + rgb_x_depth':
            self.ik_model = AllCrossDepthViTMLP(
                out_size=out_size, 
                pretrained=self.config['model'].get('pretrained', True),
                img_height=self.config['image']['resize'],
                img_width=self.config['image']['resize'],
                model_name=self.config['model']['encoder_type']
            ).to(self.device)
        elif model_type == 'depth_vit':
            self.ik_model = DepthViTMLP(
                out_size=out_size, 
                pretrained=self.config['model'].get('pretrained', True),
                img_height=self.config['image']['resize'],
                img_width=self.config['image']['resize'],
                model_name=self.config['model']['encoder_type']
            ).to(self.device)
        elif model_type == 'rgb_x_depth':
            self.ik_model = CrossDepthViTMLP(
                out_size=out_size, 
                pretrained=self.config['model'].get('pretrained', True),
                img_height=self.config['image']['resize'],
                img_width=self.config['image']['resize'],
                model_name=self.config['model']['encoder_type']
            ).to(self.device)
        elif model_type == 'depth_x_rgb':
            self.ik_model = DepthCrossRGBViTMLP(
                out_size=out_size, 
                pretrained=self.config['model'].get('pretrained', True),
                img_height=self.config['image']['resize'],
                img_width=self.config['image']['resize'],
                model_name=self.config['model']['encoder_type']
            ).to(self.device)
        elif model_type == 'resnet50':
            self.ik_model = ResNet50MLP(out_size=out_size).to(self.device)
        elif model_type == 'cnn':
            self.ik_model = CNNMLP(out_size=out_size).to(self.device)
        elif model_type == 'resnet18':
            self.ik_model = ResNet18MLP(out_size=out_size).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.ik_model.load_state_dict(torch.load(self.config['ik_model_path']))
        self.ik_model.eval()
        
    def _setup_tasks(self):
        """Get list of tasks to evaluate"""
        benchmark_dict = benchmark.get_benchmark_dict()
        suite = benchmark_dict[self.config['task']['suite_name']]()
        available_tasks = suite.get_task_names()
        
        # Filter tasks
        if self.config['task']['selected_tasks']:
            self.selected_tasks = {
                f"task{i+1}": task_name 
                for i, task_name in enumerate(available_tasks) 
                if f"task{i+1}" in self.config['task']['selected_tasks']
            }
        else:
            # If none specified, use all available tasks
            self.selected_tasks = {
                f"task{i+1}": task_name 
                for i, task_name in enumerate(available_tasks)
            }
    
    def _get_proprio_state(self, obs):
        """Get robot proprioceptive state"""
        try:
            gripper_pos = obs["robot0_gripper_qpos"]
            if isinstance(gripper_pos, np.ndarray):
                gripper_value = gripper_pos.flatten()[0:1]
            else:
                gripper_value = np.array([float(gripper_pos)]).reshape(1,)
                
            return np.concatenate([
                obs["robot0_eef_pos"].reshape(3,),
                obs["robot0_eef_quat"].reshape(4,),
                gripper_value
            ], axis=0)
        except Exception as e:
            print("Error processing gripper position:", e)
            print("Gripper position data:", obs["robot0_gripper_qpos"])
            raise
    
    def _compute_pid_control(self, goal_out, proprio_st, prev_error, integral_error):
        """Compute PID control signal"""
        # Calculate position control
        control_trans = goal_out[:3] - proprio_st[:3]
        
        # Calculate rotation control
        control_quat = (st.Rotation.from_quat(goal_out[3:7]) * 
                        st.Rotation.from_quat(proprio_st[3:7]).inv())
        control_rot = control_quat.as_euler("xyz", degrees=False)
        
        # Calculate gripper control
        gripper_goal = goal_out[7] if goal_out[7] > self.config['pid']['grip_threshold'] else 0.00
        control_gripper = (gripper_goal - proprio_st[7]) * 0.04
        
        # Combine control signals
        ik_act = np.zeros(7)
        ik_act[:3] = self.config['pid']['pos_scale'] * control_trans
        ik_act[3:6] = self.config['pid']['rot_scale'] * control_rot
        ik_act[6] = self.config['pid']['grip_scale'] * control_gripper
        ik_act = ik_act.astype(np.float32)
        
        # Apply PID control separately
        control_signal_trans = (self.config['pid']['kp_trans'] * ik_act[:3] + 
                        self.config['pid']['ki_trans'] * integral_error[:3] + 
                        self.config['pid']['kd_trans'] * (ik_act[:3] - prev_error[:3]))
        control_signal_rot = (self.config['pid']['kp_rot'] * ik_act[3:6] + 
                        self.config['pid']['ki_rot'] * integral_error[3:6] + 
                        self.config['pid']['kd_rot'] * (ik_act[3:6] - prev_error[3:6]))
        control_signal_grip = np.array([(self.config['pid']['kp_grip'] * ik_act[6] + 
                        self.config['pid']['ki_grip'] * integral_error[6] + 
                        self.config['pid']['kd_grip'] * (ik_act[6] - prev_error[6]))])
        
        control_signal = np.concatenate([control_signal_trans.reshape(-1), 
                                        control_signal_rot.reshape(-1),
                                        control_signal_grip.reshape(-1)])
        
        return control_signal, ik_act
    
    def _generate_video(self, task_prompt, side_view):
        """Generate video prediction"""
        t_before_gen = timer.perf_counter()
        
        video_clip = self.video_generator(task_prompt, side_view)
        video_clip = rearrange(video_clip, "b (f c) h w -> (b f) c h w", c=self.channel_num)
        video_clip_np = (video_clip[:, :3, :, :].permute(0, 2, 3, 1).detach().cpu().numpy()*255).astype(np.uint8)
        
        video_gen_time = timer.perf_counter() - t_before_gen
        
        return video_clip, video_clip_np, video_gen_time
    
    def _predict_goal(self, goal_img):
        """Predict goal state using IK model"""
        t_before_ik = timer.perf_counter()
        
        if self.depth_transform is not None:
            goal_img_depth = goal_img[3:, :, :]
            goal_img_rgb = goal_img[:3, :, :]
            goal_img_depth = self.depth_transform(goal_img_depth.to(self.device)).to(self.device).unsqueeze(0)
            goal_img_rgb = self.transform(goal_img_rgb.to(self.device)).unsqueeze(0)
            goal_img = torch.cat([goal_img_rgb, goal_img_depth], dim=1)
        else:
            goal_img = self.transform(goal_img.to(self.device)).unsqueeze(0)
        
        with torch.no_grad():
            goal_out = self.ik_model(goal_img)
        goal_out = goal_out.squeeze().cpu().numpy()
        
        ik_time = timer.perf_counter() - t_before_ik
        
        return goal_out, ik_time
    
    def _run_single_test(self, task_name, test_time, task_output_dir):
        """Run a single test case and return results"""
        print(f"Test {test_time+1}/{self.config['num_test_pr_task']} Start!")
        
        # Setup environment
        print(f"Using Rendering GPU: {self.config['render_gpu_id']}")
        env, task_prompt = set_up_libero_envs(
            suite_name=self.config['task']['suite_name'], 
            task_name=task_name, 
            render_device=self.config['render_gpu_id'],
            horizon=self.config['task']['horizon'],
            init_state_id=test_time
        )
        
        # Initialize
        for _ in range(self.config['init_steps']):
            obs, _, done, _ = env.step(np.zeros(7))
        
        roll_out_video = []
        success = False
        inference_times = {'video_gen': [], 'ik_model': []}
        
        # Try each video sample
        for video_sample in range(self.config['num_video_samples']):
            visual_obs, _ = process_obs(obs, extra_state_keys=[], device=self.device)
            side_view = visual_obs[:,0]
            
            # Generate video
            video_clip, video_clip_np, video_gen_time = self._generate_video(task_prompt, side_view)
            inference_times['video_gen'].append(video_gen_time)
            # Execute actions for each frame
            success, frames, ik_times, obs = self._execute_video_actions(
                env, obs, video_clip, video_clip_np, self.config['video']['act_horizon']
            )
            inference_times['ik_model'].extend(ik_times)
            roll_out_video.extend(frames)
            
            if success:
                break
        
        # Save results
        status = "success" if success else "fail"
        self._save_test_results(task_output_dir, test_time, status, roll_out_video)
        
        env.close()
        return success, inference_times
    
    def _execute_video_actions(self, env, obs, video_clip, video_clip_np, horizon):
        """Execute actions for each frame in the video"""
        success = False
        frames = []
        ik_times = []
        
        for index in range(min(horizon, len(video_clip))):
            # Predict goal
            goal_img = video_clip[index]
            goal_out, ik_time = self._predict_goal(goal_img)
            ik_times.append(ik_time)
            
            # Run PID control
            success, new_frames, obs = self._run_pid_control(
                env, obs, goal_out, video_clip_np[index]
            )
            frames.extend(new_frames)
            
            if success:
                break
                
            time.sleep(self.config['loop_sleep_time'])
        
        return success, frames, ik_times, obs
    
    def _run_pid_control(self, env, obs, goal_out, goal_img_np):
        """Run PID control loop for a single goal"""
        frames = []
        proprio_st = self._get_proprio_state(obs)
        pid_step = 0
        prev_error = np.zeros(7)
        integral_error = np.zeros(7)
        
        while (np.linalg.norm(proprio_st - goal_out) > self.config['pid']['convergence_threshold'] and 
               pid_step < self.config['pid_max_steps']):
            pid_step += 1
            
            # Calculate and apply control
            control_signal, ik_act = self._compute_pid_control(
                goal_out, proprio_st, prev_error, integral_error)
            obs, _, done, _ = env.step(control_signal)
            
            # Update state and errors
            proprio_st = self._get_proprio_state(obs)
            prev_error = ik_act
            integral_error += ik_act
            
            # Save frame
            frame = np.concatenate([cv2.flip(obs["agentview_image"], 0), goal_img_np], axis=1)
            frames.append(frame)
            
            if done:
                return True, frames, obs
            
            time.sleep(self.config['loop_sleep_time'])
        
        return False, frames, obs
    
    def _save_test_results(self, task_output_dir, test_time, status, roll_out_video):
        """Save test results as GIF and log"""
        # Save GIF
        video_name = f"test_{test_time:02d}_{status}_steps{len(roll_out_video)}.gif"
        video_path = os.path.join(task_output_dir, video_name)
        
        with imageio.get_writer(video_path, mode='I', duration=self.config['video']['duration']) as writer:
            for frame in roll_out_video:
                writer.append_data(frame)
        
        # Log result
        info_path = os.path.join(task_output_dir, "task_info.txt")
        with open(info_path, "a") as f:
            f.write(f"Test {test_time}: {'Success' if status == 'success' else 'Fail'}\n")
    
    def evaluate_task(self, task_id, task_name):
        """Evaluate a single task"""
        print(f"\nEvaluating {task_id}: {task_name}")
        task_output_dir = os.path.join(self.base_output_dir, task_id)
        os.makedirs(task_output_dir, exist_ok=True)
        
        task_successes = 0
        task_inference_times = {'video_gen': [], 'ik_model': []}
        
        # Run all test cases
        for test_time in range(self.config['num_test_pr_task']):
            success, inference_times = self._run_single_test(
                task_name, test_time, task_output_dir
            )
            
            if success:
                task_successes += 1
                
            # Collect timing data
            task_inference_times['video_gen'].extend(inference_times['video_gen'])
            task_inference_times['ik_model'].extend(inference_times['ik_model'])
        
        # Calculate and save results
        task_success_rate = (task_successes / self.config['num_test_pr_task']) * 100
        self._save_task_summary(task_output_dir, task_name, task_success_rate, 
                               task_successes, task_inference_times)
        
        return {
            'success_rate': task_success_rate,
            'successes': task_successes,
            'timing': task_inference_times
        }
        
    def _save_task_summary(self, task_output_dir, task_name, success_rate, 
                          successes, inference_times):
        """Save task summary results"""
        with open(os.path.join(task_output_dir, "task_results.txt"), "w") as f:
            f.write(f"Task: {task_name}\n")
            f.write(f"Success rate: {success_rate:.2f}%\n")
            f.write(f"Successful attempts: {successes}/{self.config['num_test_pr_task']}\n")
            f.write(f"kp_trans={self.config['pid']['kp_trans']}, ki_trans={self.config['pid']['ki_trans']}, "
                   f"kd_trans={self.config['pid']['kd_trans']}\n")
            f.write(f"kp_rot={self.config['pid']['kp_rot']}, ki_rot={self.config['pid']['ki_rot']}, "
                   f"kd_rot={self.config['pid']['kd_rot']}\n")
            f.write(f"kp_grip={self.config['pid']['kp_grip']}, ki_grip={self.config['pid']['ki_grip']}, "
                   f"kd_grip={self.config['pid']['kd_grip']}\n")
            f.write("\nInference Times:\n")
            f.write(f"Video Generator - Avg: {np.mean(inference_times['video_gen'])*1000:.2f}ms, "
                   f"Std: {np.std(inference_times['video_gen'])*1000:.2f}ms\n")
            f.write(f"IK Model - Avg: {np.mean(inference_times['ik_model'])*1000:.2f}ms, "
                   f"Std: {np.std(inference_times['ik_model'])*1000:.2f}ms\n")
    
    def run_evaluation(self):
        """Run evaluation in parallel using process pool"""
        print("Starting parallel evaluation...")
        
        # Prepare task list
        tasks = list(self.selected_tasks.items())
        
        # Create output directories for each task
        for task_id, _ in tasks:
            task_output_dir = os.path.join(self.base_output_dir, task_id)
            os.makedirs(task_output_dir, exist_ok=True)
        
        # Set number of processes, keep it low to avoid resource contention
        num_processes = self.num_processes
        print(f"Using {num_processes} processes for parallel evaluation")
        
        # Use process pool for parallel execution
        with mp.Pool(processes=num_processes) as pool:
            # Execute tasks in parallel
            results = pool.starmap(self._evaluate_task_wrapper, tasks)
        
        # Process results
        all_task_results = {}
        inference_times = defaultdict(list)
        
        for (task_id, _), result in zip(tasks, results):
            all_task_results[task_id] = result['success_rate']
            inference_times[task_id] = result['timing']
        
        # Save overall results
        self._save_overall_results(all_task_results, inference_times)
        
        return all_task_results
    
    def _evaluate_task_wrapper(self, task_id, task_name):
        """Wrapper function for task evaluation in multiprocessing"""
        print(f"\nProcess {os.getpid()} starting evaluation of {task_id}: {task_name}")
        
        # Set environment variables to avoid rendering conflicts
        os.environ['MUJOCO_GL'] = 'osmesa'
        
        # Create output directory for this task
        task_output_dir = os.path.join(self.base_output_dir, task_id)
        
        try:
            # Run all test cases
            task_successes = 0
            task_inference_times = {'video_gen': [], 'ik_model': []}
            
            for test_time in range(self.config['num_test_pr_task']):
                print(f"Process {os.getpid()}: {task_id} test {test_time+1}/{self.config['num_test_pr_task']}")
                
                success, inference_times = self._run_single_test(
                    task_name, test_time, task_output_dir
                )
                
                if success:
                    task_successes += 1
                    
                # Collect timing data
                task_inference_times['video_gen'].extend(inference_times['video_gen'])
                task_inference_times['ik_model'].extend(inference_times['ik_model'])
            
            # Calculate and save results
            task_success_rate = (task_successes / self.config['num_test_pr_task']) * 100
            self._save_task_summary(task_output_dir, task_name, task_success_rate, 
                                  task_successes, task_inference_times)
            
            print(f"Process {os.getpid()} completed task {task_id}")
            
            return {
                'success_rate': task_success_rate,
                'successes': task_successes,
                'timing': task_inference_times
            }
            
        except Exception as e:
            print(f"Process {os.getpid()} encountered error while evaluating task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success_rate': 0,
                'successes': 0,
                'timing': {'video_gen': [], 'ik_model': []}
            }

    def _save_overall_results(self, all_task_results, inference_times):
        """Save overall evaluation results"""
        overall_success_rate = sum(all_task_results.values()) / len(all_task_results)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        with open(os.path.join(self.base_output_dir, "overall_results.txt"), "w") as f:
            f.write(f"Experiment Timestamp: {timestamp}\n")
            if self.config.get('use_seed', False):
                f.write(f"Random Seed: {self.config['seed']}\n")
            f.write(f"Overall success rate across all tasks: {overall_success_rate:.2f}%\n\n")
            
            # Calculate and write overall inference times
            all_video_gen_times = [time for task_times in inference_times.values() 
                                  for time in task_times['video_gen']]
            all_ik_times = [time for task_times in inference_times.values() 
                            for time in task_times['ik_model']]
            
            f.write("Overall Inference Times:\n")
            f.write(f"Video Generator - Avg: {np.mean(all_video_gen_times)*1000:.2f}ms, "
                    f"Std: {np.std(all_video_gen_times)*1000:.2f}ms\n")
            f.write(f"IK Model - Avg: {np.mean(all_ik_times)*1000:.2f}ms, "
                    f"Std: {np.std(all_ik_times)*1000:.2f}ms\n\n")
            
            f.write("Individual Task Results:\n")
            for task_id, success_rate in all_task_results.items():
                f.write(f"{task_id}: {success_rate:.2f}%\n")
                
            f.write("\n")
            f.write(f"Model: {self.config['model']['type']}\n")
            f.write(f"Model path: {self.config['ik_model_path']}\n")
            f.write(f"Video generator path: {self.config['video_generator_path']}\n")
            f.write(f"Video action horizon: {self.config['video']['act_horizon']}\n")
            f.write(f"Video latent: {self.config['video']['latent']}\n")
            f.write(f"Video depth: {self.config['video']['depth']}\n")
            f.write(f"Experience name: {self.config['output']['exp_name']}\n")


def main():
    evaluator = IKEvaluator()
    results = evaluator.run_evaluation()
    print("Evaluation completed, results:", results)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()