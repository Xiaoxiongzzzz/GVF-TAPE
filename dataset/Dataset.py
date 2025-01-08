import torch
from torch.utils.data import Dataset
import h5py
import torch.nn.functional as F
from torchvision import transforms
import os
import random
class LiberoDataset(Dataset):
    def __init__(self, task_dataset_path, diffusion_image_size=(48,64), num_frame_stack=None, extra_state_keys=None, augmentation=True):
        '''
        Args:
        task_dataset_path: path to task dataset hdf5 (e.g. libero_spatial, libero_goal)
        diffusion_image_size: the size of image to input to video generation model (e.g. (48, 64))
        num_frame_stack: how many previous obs are used (e.g. 10)
        extra_state_keys: name of used state (e.g. ["joint_states", "gripper_states"])
        '''
        super().__init__()
        self.diffusion_image_size = diffusion_image_size
        self.num_frame_stack = num_frame_stack
        self.extra_state_keys = extra_state_keys
        self.task_dataset_path = task_dataset_path
        self.augmentation = augmentation

        self.demos = []
        self.demos_horizon = []
        with h5py.File(task_dataset_path, 'r') as f:
            self.demos = list(f['data'].keys())
            for demo in self.demos:
                self.demos_horizon.append(len(f['data'][demo]['actions']))
        self.length = sum(self.demos_horizon)
        print(f"The number of demos is:{len(self.demos)}, the number of data is: {self.length}")

        self.resize_diffusion_image = transforms.Resize(self.diffusion_image_size)
        if self.augmentation:
            self.augmentator = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.RandomCrop(size=(128,128), padding=4, fill=0, padding_mode='constant')
            ])
    def __getitem__(self, index):
        '''
        Args:
        Given a index, return the visual_obs, extra_state, action

        Return:
        diffusion_image: (num_frame_stack, 3, **diffusion_imagesize)
        visual_obs: (views, num_frame_stack, 3, 128, 128)
        extra_state: {k: (num_frame_stack, n)}
        action: (num_frame_stack ,7,)    
        task_text: str

        Note: NOT NORMALIZED!!!(0-255)
        '''
        demo_index, time_index = self.from_index_to_demo_and_time(index)

        with h5py.File(self.task_dataset_path, 'r') as f:
            demo = f['data'][self.demos[demo_index]]
            if time_index >= self.num_frame_stack:
                agentview = torch.from_numpy(demo['obs']['agentview_rgb'][time_index-self.num_frame_stack+1:time_index+1])    # (num_frame_stack, height, width, channel)
                eye_in_hand = torch.from_numpy(demo['obs']['eye_in_hand_rgb'][time_index-self.num_frame_stack+1:time_index+1])  # (num_frame_stack, height, width, channel)
                visual_obs = torch.stack([agentview, eye_in_hand], dim=0)   #(views, num_frame_stack, height, width, channel)
                extra_state = {k: torch.from_numpy(demo['obs'][k][time_index-self.num_frame_stack+1:time_index+1]) for k in self.extra_state_keys} #{k:(num_frame_stack,n)}
                action = torch.from_numpy(demo['actions'][time_index-self.num_frame_stack+1:time_index+1])
            else:
                agentview = torch.from_numpy(demo['obs']['agentview_rgb'][:time_index+1])
                eye_in_hand = torch.from_numpy(demo['obs']['eye_in_hand_rgb'][:time_index+1])
                visual_obs = torch.stack([agentview, eye_in_hand], dim=0)
                extra_state = {k:torch.from_numpy(demo['obs'][k][:time_index+1]) for k in self.extra_state_keys}
                action = torch.from_numpy(demo['actions'][:time_index+1])
                visual_obs, extra_state, action = self.pad_to_same_length(visual_obs, extra_state, action)      #pad time to num frame stack with 0
        
        extra_state = {k: v.to(torch.float32) for k, v in extra_state.items()}
        action = action.to(torch.float32)

        visual_obs = torch.flip(visual_obs, dims=[-3])     #Vertical Flip the image (views, num_frame_stack, height, width, channel)
        visual_obs = visual_obs.permute(0, 1, 4, 2, 3)      #(views, num_frame_stack, channel, height, width)
        diffusion_image = self.resize_diffusion_image(visual_obs[0])

        if self.augmentation:
            visual_obs = (self.augmentator(visual_obs/255.0)*255).int()
        task_text = self.get_task_text()

        return diffusion_image, visual_obs, extra_state, action, task_text
    def pad_to_same_length(self, visual_obs, extra_state, action):
        '''
        Args:
        Given a obs and extra_state, return the padded obs and extra state
        visual_obs: (views, times, height, width, channel)
        extra_state: {k: (times, n)}
        action: (time, n)
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
        
        return visual_obs, extra_state, action
    def from_index_to_demo_and_time(self,index):
        '''
        Args:
        Given a index, return the demo index and time in the demo
        '''
        demo_index = 0
        time_index = index

        while time_index >= self.demos_horizon[demo_index]:
            time_index -= self.demos_horizon[demo_index]
            demo_index += 1

        return demo_index, time_index
    def get_task_text(self):
        task_text = os.path.basename(self.task_dataset_path).split('.')[0]
        task_text = task_text.replace('_', ' ')
        task_text = task_text.replace('demo', '')
        
        return task_text
        
    def __len__(self):
        return self.length


class LiberoSuiteDataset(Dataset):
    def __init__(self, suite_path, ratio, diffusion_image_size=(48,64), num_frame_stack=None, extra_state_keys=None, augmentation=True, return_feature=False, mode="train"):
        '''
        Args:
        suite_path: path to suite dir (e.g. libero_spatial, libero_goal)
        ratio: the demo in every task used for training(e.g. 0.2)
        diffusion_image_size: the size of image to input to video generation model (e.g. (48, 64))
        num_frame_stack: how many previous obs are used (e.g. 10)
        '''
        super().__init__()
        self.diffusion_image_size = diffusion_image_size
        self.num_frame_stack = num_frame_stack
        self.extra_state_keys = extra_state_keys
        self.suite_path = suite_path
        self.augmentation = augmentation
        self.ratio = ratio
        self.mode = mode
        self.return_feature = return_feature
        self.resize_diffusion_image = transforms.Resize(self.diffusion_image_size, antialias=True)
        if self.augmentation:
            self.augmentator = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.RandomCrop(size=(128,128), padding=4, fill=0, padding_mode='constant')
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
        diffusion_image: (num_frame_stack, 3, **diffusion_imagesize)
        visual_obs: (views, num_frame_stack, 3, 128, 128)
        extra_state: {k: (num_frame_stack, n)}
        action: (num_frame_stack ,7,)    
        task_text: str

        Note: NOT NORMALIZED!!!(0-255)
        '''
        hdf5_name, demo, time_index = self.index_codebook[index]
        with h5py.File(os.path.join(self.suite_path, hdf5_name), 'r') as f:
            demo = f['data'][demo]
            if time_index >= self.num_frame_stack:
                agentview = torch.from_numpy(demo['obs']['agentview_rgb'][time_index-self.num_frame_stack+1:time_index+1])    # (num_frame_stack, height, width, channel)
                eye_in_hand = torch.from_numpy(demo['obs']['eye_in_hand_rgb'][time_index-self.num_frame_stack+1:time_index+1])  # (num_frame_stack, height, width, channel)
                visual_obs = torch.stack([agentview, eye_in_hand], dim=0)   #(views, num_frame_stack, height, width, channel)
                extra_state = {k: torch.from_numpy(demo['obs'][k][time_index-self.num_frame_stack+1:time_index+1]) for k in self.extra_state_keys} #{k:(num_frame_stack,n)}
                action = torch.from_numpy(demo['actions'][time_index-self.num_frame_stack+1:time_index+1])
                if self.return_feature:
                    feature = torch.from_numpy(demo['feature'][time_index-self.num_frame_stack+1:time_index+1])
            else:
                agentview = torch.from_numpy(demo['obs']['agentview_rgb'][:time_index+1])
                eye_in_hand = torch.from_numpy(demo['obs']['eye_in_hand_rgb'][:time_index+1])
                visual_obs = torch.stack([agentview, eye_in_hand], dim=0)
                extra_state = {k:torch.from_numpy(demo['obs'][k][:time_index+1]) for k in self.extra_state_keys}
                action = torch.from_numpy(demo['actions'][:time_index+1])
                if self.return_feature:
                    feature = torch.from_numpy(demo['feature'][:time_index+1])          #time, channel, height, width
                    visual_obs, extra_state, action, feature = self.pad_to_same_length(visual_obs, extra_state, action, feature)      #pad time to num frame stack with 0
                else:
                    visual_obs, extra_state, action, _ = self.pad_to_same_length(visual_obs, extra_state, action)      #pad time to num frame stack with 0
            task_embed = torch.from_numpy(demo['task_embed'][:]).squeeze(0)

        extra_state = {k: v.to(torch.float32) for k, v in extra_state.items()}
        action = action.to(torch.float32)
        if self.return_feature:
            feature = feature.to(torch.float32)

        visual_obs = torch.flip(visual_obs, dims=[-3])     #Vertical Flip the image (views, num_frame_stack, height, width, channel)
        visual_obs = visual_obs.permute(0, 1, 4, 2, 3)      #(views, num_frame_stack, channel, height, width)
        diffusion_image = self.resize_diffusion_image(visual_obs[0])

        if self.augmentation:
            visual_obs = (self.augmentator(visual_obs/255.0)*255).int()
        task_text = self.get_task_text(hdf5_name)
        
        if self.return_feature:
            return visual_obs, extra_state, action, task_text, feature
        else:
            return diffusion_image, visual_obs, extra_state, action, task_text, task_embed
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