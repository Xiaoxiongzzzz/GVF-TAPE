from dataset.Dataset4IK import LiberoSuiteDataset4IK
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np

extra_state_keys = ["ee_states", "gripper_states"]
total_dataset = LiberoSuiteDataset4IK(
    suite_path="/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero/libero_spatial",
    ratio=1,
    extra_state_keys=extra_state_keys,
    augmentation=False,
    mode="train",
)
data_loader = DataLoader(total_dataset, batch_size=512, shuffle=True, num_workers=8)
batch_sum, batch_squared_sum, num_batches = np.zeros(8), np.zeros(8), 0
for batch in tqdm(data_loader):
    obs_goal, _, _, _ = batch
    pose = obs_goal["ee_states"].numpy()

    batch_sum += np.mean(pose, axis=0)
    batch_squared_sum += np.mean(pose**2, axis=0)
    num_batches += 1

mean = batch_sum / num_batches
std = np.sqrt(batch_squared_sum / num_batches - mean**2)

print(f"mean is {mean}, std is {std}")
