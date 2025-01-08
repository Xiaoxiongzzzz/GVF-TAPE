from policy.ik_model.resnet import ResNet18, ResNet50
from policy.ik_model.rflow_ik import FlowIKModel
from utils.env_utils import set_up_libero_envs
from libero.libero import benchmark
from torchvision.utils import save_image
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import h5py 
import imageio
import cv2
import os
import tqdm
def main():
    device = torch.device("cuda")
    resnet = ResNet50(input_dim=3, output_dim=6).to(device)
    resnet.load_state_dict(torch.load("/mnt/home/ZhangXiaoxiong/Documents/VideoGeneration/results/rf_ik/model_95.pt"))
    resnet.eval()

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_name = "libero_spatial"
    suite = benchmark_dict[suite_name]()
    task_names = suite.get_task_names()
    task_name = task_names[0]
    env, task_prompt = set_up_libero_envs(suite_name, task_name, 1)
    print(f"Now env name is: \n {task_prompt}")
    obs = env.reset()
    video_clip = []
    text_list = []
    with h5py.File("/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5") as f:
        images = f["data"]["demo_0"]["obs"]["agentview_rgb"][:]
        demo = f["data"]["demo_0"]
        with torch.no_grad():
            with tqdm.tqdm(range(images.shape[0])) as pbar:
                for i, goal_image in enumerate(images):
                    x_goal = process_image(goal_image).to(device)
                    ee_state_goal = resnet(x_goal).detach().cpu().numpy()[0]
                    ground_truth = np.round(demo["obs"]["ee_states"][i], 2)
                    text_list.append(f"GT:{ground_truth}, Pred:{np.round(ee_state_goal, 2)}")
                    for _ in range(10):
                        translation = ee_state_goal[:3] - obs["robot0_eef_pos"]

                        orientation_current = R.from_quat(obs["robot0_eef_quat"])
                        orientation_goal = R.from_euler('xzy', ee_state_goal[-3:], degrees=False)
                        rotation = rotation_control(orientation_goal, orientation_current)

                        control = np.concatenate([translation*5, rotation*5, [0]], axis=0)
                        obs, _, _, _ = env.step(control)
                        
                        video_clip.append(np.concatenate([cv2.flip(obs["agentview_image"], 0), cv2.flip(goal_image, 0)], axis=1))
                    pbar.update(1)
        env.close()
    os.makedirs("./results/ik_policy", exist_ok=True)

    with imageio.get_writer("./results/ik_policy/demo.gif", mode='I', duration=100) as writer:
        for image in video_clip:
            writer.append_data(image)
def process_image(image):
    image = torch.flip(torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)/255.0, dims=[-2]).to(torch.float32)
    return image
def rotation_control(orientation_goal, orientation_current):
    """Get SO(3) input and return 'xyz' delta Euler angle"""
    rotation_operation = orientation_goal * orientation_current.inv()
    rotation_operation = rotation_operation.as_euler('xyz', degrees=False)
    return rotation_operation

if __name__ == "__main__":
    main()