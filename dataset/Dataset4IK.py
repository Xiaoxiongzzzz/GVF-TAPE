from torch.utils.data import Dataset
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import torch
import h5py
import os
import random
class LiberoSuiteDataset4IK(Dataset):
    def __init__(self, suite_path, ratio, extra_state_keys=None, augmentation=True, mode="train"):
        '''
        Args:
        suite_path: path to suite dir (e.g. libero_spatial, libero_goal)
        ratio: the demo in every task used for training(e.g. 0.2)
        num_frame_stack: how many previous obs are used (e.g. 10)
        '''
        super().__init__()
        self.extra_state_keys = extra_state_keys
        self.suite_path = suite_path
        self.augmentation = augmentation
        self.ratio = ratio
        self.mode = mode
        
        self.mean = torch.Tensor([-0.02596965, 0.10603805, 1.05615763, 0.96777721, -0.04932071, -0.03729568, 0.04566995, 0.01967621])
        self.std = torch.Tensor([0.11212396, 0.1382502, 0.10310984, 0.05012393, 0.19773889, 0.11159955, 0.05857035, 0.0170986])

        if self.augmentation:
            self.augmentator = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                # transforms.RandomCrop(size=(128,128), padding=4, fill=0, padding_mode='constant')
                # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ])

        self.hdf5_files = self.get_hdf5_files(suite_path)
        self.train_demos = self.split_demos(self.hdf5_files, self.ratio)
        self.demos_length = self.get_demos_length(self.train_demos, self.hdf5_files)
        self.index_codebook = self.get_index_codebook()
        print(f"The number of demos is {sum([len(demos) for demos in self.train_demos])}")
    def __getitem__(self, index):
        '''
        Args:
        Given a index, return the visual_obs, extra_state, action

        Return:
        side_view_goal: (3, 128, 128)
        visual_obs: (views, 3, 128, 128)
        extra_state: {k: (n)}
        action: (7,)    
        task_text: str

        Note: NORMALIZED IMAGE!!!(0-1)
        '''
        hdf5_name, demo, time_index = self.index_codebook[index]
        if time_index == 0:
            time_index += 1
        current_index = max(0, random.randint(time_index-4, time_index))

        with h5py.File(os.path.join(self.suite_path, hdf5_name), 'r') as f:
            demo = f['data'][demo]
            side_view_goal = torch.flip(torch.from_numpy(demo['obs']['agentview_rgb'][time_index]), dims=[-3]).to(torch.float32)/255.0      #height, width, channel
            side_view_current = torch.flip(torch.from_numpy(demo['obs']['agentview_rgb'][current_index]), dims=[-3]).to(torch.float32)/255.0      #height, width, channel
            eye_in_hand_current = torch.flip(torch.from_numpy(demo['obs']['eye_in_hand_rgb'][current_index]), dims=[-3]).to(torch.float32)/255.0
            observation_current = {k: torch.from_numpy(demo['obs'][k][current_index]).to(torch.float32) for k in self.extra_state_keys}     #{k:(n)}
            observation_goal = {k: torch.from_numpy(demo['obs'][k][time_index]).to(torch.float32) for k in self.extra_state_keys}     #{k:(n)}
            action_current = torch.from_numpy(demo['actions'][current_index]).to(torch.float32)

        visual_obs = torch.stack([side_view_current, eye_in_hand_current], dim=0).permute(0, 3, 1, 2)        #(views, channel, height, width)
        side_view_goal = side_view_goal.permute(2, 0, 1)    #(channel, height, width)

        if self.augmentation:
            visual_obs = (self.augmentator(visual_obs))

        observation_goal["side_view"] = side_view_goal
        observation_goal["ee_states"] = (torch.cat([observation_goal["ee_states"][:3],
                                                torch.from_numpy(R.from_euler('xzy', observation_goal["ee_states"][3:], degrees=False).as_quat()), 
                                                observation_goal["gripper_states"][:1]],
                                                dim=0)-self.mean)/self.std
                                                     
        observation_current["side_view"] = visual_obs[0]
        observation_current["eye_in_hand"] = visual_obs[1]
        task_text = self.get_task_text(hdf5_name)

        return observation_goal, observation_current, action_current, task_text

    def __len__(self):
        return len(self.index_codebook)
    
    def pad_to_same_length(self, visual_obs, extra_state, action, feature=None):
        '''
        Args:
        Given a obs and extra_state, return the padded obs and extra state
        visual_obs: (views, times, height, width, channel)
        extra_state: {k: (times, n)}
        action: (time, n)
        feature: (time, channel, height, width)
        '''
        # pad visual obs to num frame stack
        visual_obs = F.pad(visual_obs, 
                           (0, 0, 0, 0, 0, 0, self.num_frame_stack-visual_obs.shape[1], 0),
                           mode='constant',
                           value=0,)
        extra_state = {k: F.pad(extra_state[k],
                                (0, 0, self.num_frame_stack-extra_state[k].shape[0], 0),
                                mode='constant',
                                value=0,) for k in extra_state.keys()}
        action = F.pad(action,
                       (0, 0, self.num_frame_stack-action.shape[0], 0),
                       mode='constant',
                       value=0,)
        
        if feature is not None:
            feature = F.pad(feature,
                            (0, 0, 0, 0, 0, 0, self.num_frame_stack-feature.shape[0], 0),
                            mode='constant',
                            value=0,)
        
        return visual_obs, extra_state, action, feature
    def get_index_codebook(self):
        index_codebook = []
        for i, hdf5_name in enumerate(self.hdf5_files):
            for j, demo in enumerate(self.train_demos[i]):
                index_codebook.extend([hdf5_name, demo, t] for t in range(self.demos_length[i][j]))
        
        return index_codebook
    def get_task_text(self, file_name: str) -> str:
        '''
        Args:
            file_name: e.g. "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
        '''
        task_prompt = file_name.split(".")[0]
        task_prompt = task_prompt.replace("_", " ")
        task_prompt = task_prompt[:-5]

        return task_prompt
    
    def get_hdf5_files(self, suite_path: str) -> list:
        ''''
        Args:
            suite_path: path to the suite dir
        Return:
            hdf5_list: [hdf5_name, ] 
        '''
        hdf5_list = []
        for file in os.listdir(suite_path):
            if file.endswith(".hdf5"):
                hdf5_list.append(file)
        
        return hdf5_list
    def split_demos(self, hdf5_list: list, ratio: float) -> list:
        '''
        Args:
            hdf5_list: [hdf5_name, ...]
            ratio: the demo in every task used for training(e.g. 0.2)
        Return:
            trian_demos: [[task1_demos], [task2_demos], ...]
        '''
        train_demos = []
        for hdf5 in hdf5_list:
            with h5py.File(os.path.join(self.suite_path, hdf5), 'r') as f:
                demo_list = list(f['data'].keys())
                if self.mode == "train":
                    random_demos = demo_list[:int(len(demo_list)*ratio)]
                else:
                    random_demos = demo_list[-int(len(demo_list)*ratio):]
                train_demos.append(random_demos)

        return train_demos
    def get_demos_length(self, train_demos: list, hdf5_list: list) -> list:
        '''
        Args:
            train_demos: [[task1_demos], [task2_demos],...]
            hdf5_list: [hdf5_name,...]
        Return:
            demos_length: [[task1_demos_length], [task2_demos_length],...]
        '''
        demos_length = []
        for i in range(len(hdf5_list)):
            
            task_demo_length = []
            
            with h5py.File(os.path.join(self.suite_path, hdf5_list[i]), 'r') as f:
                for demo in train_demos[i]:
                    task_demo_length.append(len(f['data'][demo]['actions']))
            
            demos_length.append(task_demo_length)
        return demos_length