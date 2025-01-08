from libero.libero import benchmark
from policy.vilt import BCViLTPolicy
from utils.policy_utils import prepare_feature_extractor,replace_checkpoint_keys,set_seed
from utils.env_utils import set_up_libero_envs
from utils.env_utils import process_obs
from omegaconf import OmegaConf
from ema_pytorch import EMA
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import os
import cv2
import numpy as np
import torch
import yaml
import imageio
import argparse
import multiprocessing as mp

def run_task(task_name, suite_name, cfg, checkpoint_path, render_device, rollout_times, rollout_horizon, is_replace_checkpoint_keys):
    device = torch.device("cuda")
    set_seed(0)
    ##Step 1. Initialize the feature extractor model and policy model
    feature_extractor = prepare_feature_extractor(model_path=cfg.video_generation_model_path,
                                                  device=device)
    feature_extractor.eval()
    policy = BCViLTPolicy(**cfg.model_cfg).to(device)
    state_dict = torch.load(checkpoint_path)
    if is_replace_checkpoint_keys:
        state_dict = replace_checkpoint_keys(state_dict)
    policy.load_state_dict(state_dict)
    policy.eval()

    ##Step 2. Set up the environment
    env, task_prompt = set_up_libero_envs(suite_name, task_name, render_device)

    success_list = []
    for time in tqdm(range(rollout_times), desc=f"Task: {task_name}"):
        image_list = []
        obs = env.reset()
        policy.reset(device)

        for step in range(rollout_horizon):
            visual_obs, diffusion_image, extra_states = process_obs(
                obs, cfg.extra_state_keys, cfg.diffusion_image_size, device
            )
            feature = feature_extractor(diffusion_image, task_prompt)
            # feature = torch.randn(1, 1, 1120, 6, 48, 64).to(device)
            action = policy.act(visual_obs, feature, extra_states)
            obs, _, done, _ = env.step(action[0])

            image_list.append(cv2.flip(obs['agentview_image'], 0))

            if done:
                break
        success_list.append(done)

        output_dir = f"./results/policy/{task_name}"
        os.makedirs(output_dir, exist_ok=True)
        imageio.mimsave(f"{output_dir}/video{time}.gif", image_list, duration=20)
    
    env.close()
    success_rate = np.mean(success_list)*100
    print(f"Task {task_name}: Success rate = {success_rate}")

    return task_name, success_rate
def main(args):
    # Set up the hyperpatameters we need
    mp.set_start_method('spawn', force=True)
    cfg = yaml.safe_load(open('/mnt/home/ZhangXiaoxiong/Documents/VideoGeneration/conf/libero_vilt.yaml'))
    cfg = OmegaConf.create(cfg)

    rollout_times = 20
    rollout_horizon = 600
    benchmark_dict = benchmark.get_benchmark_dict()
    suite_name = args.suite
    suite = benchmark_dict[suite_name]()
    task_names = suite.get_task_names()
    render_device = args.render_device

    # Parallel workers
    max_workers = args.max_workers if args.max_workers else 5

    task_func = partial(
        run_task,
        suite_name=suite_name,
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        render_device=render_device,
        rollout_times=rollout_times,
        rollout_horizon=rollout_horizon,
        is_replace_checkpoint_keys=True
    )
    
    success_rate = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task_func, task_name) for task_name in task_names]
        
        for future in as_completed(futures):
            try:
                task_name, rate = future.result()
                success_rate[task_name] = rate
            except Exception as e:
                print(f"Error occurred: {e}")

    os.makedirs("./results/policy", exist_ok=True)
    with open("./results/policy/result.txt", "w") as f:
        for k,v in success_rate.items():
            f.write(f"{k}: {v}%\n")
    print(f"Overall successful rate is {np.mean(list(success_rate.values()))}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the policy checkpoint.")
    parser.add_argument("--suite", type=str, required=True, help="Suite env we need to roll out. e.g. libero_spatial")
    parser.add_argument("--render-device", type=int, required=True, help="The GPU id used for rendering env")
    parser.add_argument("--max-workers", type=int, required=True, help="The num of workers need to use")
    args = parser.parse_args()
    main(args)