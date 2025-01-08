from diffusion_model.UNets import UnetBridge as Unet
from diffusion_model.GoalDiffusion import GoalGaussianDiffusion
from policy.bc_vilt import VanillaBCViLTPolicy
from dataset.Dataset import LiberoDataset,LiberoSuiteDataset
from diffusion_model.FeatureExtractor import FeatureExtractor
from utils.policy_utils import prepare_feature_extractor

import argparse
import wandb
import yaml
import os
import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import random_split
from ema_pytorch import EMA
def main(args):
    ## Step 1. Set up the hyperpatameters we need
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_wandb = args.use_wandb
    cfg = yaml.safe_load(open('/mnt/home/ZhangXiaoxiong/Documents/VideoGeneration/conf/libero_vilt.yaml'))
    cfg = OmegaConf.create(cfg)

    if use_wandb:
        wandb.init(project="avdc-guided-policy", name="vanilla-bc")

    ## Step 2. Initialize the policy model and optimizer
    policy = VanillaBCViLTPolicy(**cfg.model_cfg).to(device)
    if args.checkpoint is not None:
        policy.load_state_dict(torch.load(args.checkpoint))
        print(f"Load policy checkpoint from {args.checkpoint}")

    optimizer = optim.AdamW(policy.parameters(), lr = cfg.optimizer_cfg.lr, weight_decay=cfg.optimizer_cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    ## Step 3. Prepare the dataset and dataloader
    dataset_path = cfg.dataset_path
    train_dataset = LiberoSuiteDataset(
                            suite_path=dataset_path, 
                            ratio=cfg.train_ratio,
                            num_frame_stack=cfg.frame_stack,
                            extra_state_keys=cfg.extra_state_keys,
                            return_feature=False,
                            mode="train",
                            )

    validation_dataset = LiberoSuiteDataset(
                            suite_path=dataset_path, 
                            ratio=0.04,
                            num_frame_stack=cfg.frame_stack,
                            extra_state_keys=cfg.extra_state_keys,
                            return_feature=False,
                            augmentation=False,
                            mode="valid",
                            )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=True)

    ## Step 4. Training Loop
    for epoch in range (cfg.num_epochs):
        with tqdm(total=len(train_dataloader), desc=f"Epoch: {epoch + 1}/{cfg.num_epochs}") as pbar:
            for batch in train_dataloader:
                policy.train()
                # 1. load data to specified device
                _, visual_obs, extra_states, action, task_text, task_embed = batch
                visual_obs, action, task_embed = visual_obs.to(device), action.to(device), task_embed.to(device)
                extra_states = {k: v.to(device) for k, v in extra_states.items()}

                #2. compute the loss and log to wandb
                loss ,ret_dict = policy.forward_loss(visual_obs, extra_states, task_embed, action)
                
                if use_wandb:
                    wandb.log(ret_dict)
                    wandb.log({"epoch": epoch, "lr": optimizer.param_groups[0]['lr']})


                #3. backpropagation and update the policy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        scheduler.step()
        # 4. evaluation and save policy model
        if epoch % cfg.evaluation_every_epoch == 0:
            print(f'Validating at epoch {epoch}')
            policy.eval()
            running_loss = []
            with torch.no_grad():
                for batch in validation_dataloader:
                    _, visual_obs, extra_states, action, task_text, task_embed = batch
                    visual_obs, action, task_embed = visual_obs.to(device), action.to(device), task_embed.to(device)
                    extra_states = {k: v.to(device) for k, v in extra_states.items()}

                    loss, _ = policy.forward_loss(visual_obs, extra_states, task_embed,action)
                    running_loss.append(loss.item())
                    
                validation_loss = np.mean(running_loss)
                print(f"validation loss:{validation_loss}")
                if use_wandb:
                    wandb.log({"val_loss": validation_loss})

            os.makedirs(f"../results/{args.exp_dir}", exist_ok=True)
            torch.save(policy.state_dict(), f"../results/{args.exp_dir}/model_{epoch}.pt")

    torch.save(policy.state_dict(), f"../results/{args.exp_dir}/model_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", default=False, type=bool, help="If use wandb to log the training process")
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to the policy checkpoint")
    parser.add_argument("--exp-dir", required=True, type=str, help="Name of experiment directionary")


    args = parser.parse_args()
    main(args)