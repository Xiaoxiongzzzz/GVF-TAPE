from diffusion_model.UNets import UnetBridge as Unet
from diffusion_model.GoalDiffusion import GoalGaussianDiffusion
from policy.vilt import BCViLTPolicy
from dataset.Dataset import LiberoDataset, LiberoSuiteDataset
from diffusion_model.FeatureExtractor import FeatureExtractor
from utils.policy_utils import prepare_feature_extractor, set_seed
from accelerate import Accelerator
import time
import argparse
import wandb
import yaml
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from torch.utils.data import random_split
from ema_pytorch import EMA
def main(args):
    ## Step 1. Set up the hyperpatameters we need
    accelerator = Accelerator()
    set_seed(0)
    device = accelerator.device
    diffusion_image_size = (48, 64)
    return_feature = False
    
    use_wandb = args.use_wandb
    cfg = yaml.safe_load(open('/mnt/home/ZhangXiaoxiong/Documents/VideoGeneration/conf/libero_vilt.yaml'))
    cfg = OmegaConf.create(cfg)

    if use_wandb and accelerator.is_main_process:
        wandb.init(project="avdc-guided-policy")
    ## Step 2. Initialize the vedio generation feature extractor model
    feature_extractor = prepare_feature_extractor(model_path=cfg.video_generation_model_path, device=device)
    for param in feature_extractor.parameters():
        param.requires_grad = False

    ## Step 3. Initialize the policy model and optimizer
    policy = BCViLTPolicy(**cfg.model_cfg).to(device)
    if args.checkpoint is not None:
        policy.load_state_dict(torch.load(args.checkpoint))
        print(f"Load policy checkpoint from {args.checkpoint}")

    optimizer = optim.AdamW(policy.parameters(), lr = cfg.optimizer_cfg.lr, weight_decay=cfg.optimizer_cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs*accelerator.num_processes)

    ## Step 4. Prepare the dataset and dataloader
    dataset_path = cfg.dataset_path
    train_dataset = LiberoSuiteDataset(
                            suite_path=dataset_path, 
                            ratio=cfg.train_ratio,
                            diffusion_image_size=diffusion_image_size, 
                            num_frame_stack=cfg.frame_stack,
                            extra_state_keys=cfg.extra_state_keys,
                            return_feature=return_feature,
                            )
    validation_dataset = LiberoSuiteDataset(
                            suite_path=dataset_path, 
                            ratio=0.04,
                            diffusion_image_size=diffusion_image_size, 
                            num_frame_stack=cfg.frame_stack,
                            extra_state_keys=cfg.extra_state_keys,
                            return_feature=return_feature,
                            mode="validation"
                            )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    feature_extractor, policy, optimizer, scheduler, train_dataloader, validation_dataloader = accelerator.prepare(
                            feature_extractor, policy, optimizer, scheduler, train_dataloader, validation_dataloader,
                            )
    ## Step 5. Training Loop
    for epoch in range (cfg.num_epochs):
        with tqdm(total=len(train_dataloader), desc=f"Epoch: {epoch + 1}/{cfg.num_epochs}", disable=not accelerator.is_main_process) as pbar:
            policy.train()
            for batch in train_dataloader:
                # 1. load data to specified device
                diffusion_image, visual_obs, extra_states, action, task_text, _= batch
                diffusion_image, visual_obs, action = diffusion_image.to(device), visual_obs.to(device), action.to(device)
                extra_states = {k: v.to(device) for k, v in extra_states.items()}
                feature = feature_extractor(diffusion_image, task_text)
                #2. compute the loss and log to wandb
                pred_action = policy(visual_obs, feature, extra_states)
                loss = F.mse_loss(pred_action, action, reduction="mean")
                if use_wandb and accelerator.is_main_process:
                    wandb.log({"loss": loss.item(), "epoch": epoch, "lr": optimizer.param_groups[0]['lr']})

                #3. backpropagation and update the policy
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        scheduler.step()
        # 4. evaluation and save policy model
        if epoch % cfg.evaluation_every_epoch == 0:
            if accelerator.is_main_process:
                print(f'Validating at epoch {epoch}')
            policy.eval()
            running_loss = []
            with torch.no_grad():
                for batch in validation_dataloader:
                    diffusion_image, visual_obs, extra_states, action, task_text, _ = batch
                    diffusion_image, visual_obs, action = diffusion_image.to(device), visual_obs.to(device), action.to(device)
                    extra_states = {k: v.to(device) for k, v in extra_states.items()}
                    feature = feature_extractor(diffusion_image, task_text) 

                    pred_action = policy(visual_obs, feature, extra_states)
                    loss = F.mse_loss(pred_action, action, reduction="mean")

                    running_loss.append(loss.item())
                    
            if accelerator.is_main_process:    
                validation_loss = np.mean(running_loss)
                print(f"validation loss:{validation_loss}")
                if use_wandb:
                    wandb.log({"val_loss": validation_loss})

                os.makedirs(f"../results/policy_model2", exist_ok=True)
                torch.save(accelerator.unwrap_model(policy.state_dict()), f"../results/policy_model2/model_{epoch}.pt")

    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(policy.state_dict()), "../results/policy_model2/model_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", default=False, type=bool, help="Whether use wandb to log the training process")
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to the policy checkpoint")

    args = parser.parse_args()
    main(args)