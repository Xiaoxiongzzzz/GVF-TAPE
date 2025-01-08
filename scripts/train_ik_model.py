from dataset.Dataset4IK import LiberoSuiteDataset4IK
from torchvision.utils import save_image
from policy.ik_model.resnet import ResNet50, ResNet50Pretrained
from policy.ik_model.rflow_ik import FlowIKModel
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from peft import PeftModel
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
import argparse
import wandb
import os

def main(args):
    device = torch.device("cuda")
    use_vae = False
    resnet = ResNet50Pretrained(output_dim=8).to(device)
    if use_vae:
        vae = prepare_vae(device)

    optimizer = torch.optim.AdamW(resnet.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

    extra_state_keys = ["ee_states", "gripper_states"]
    if args.use_wandb:
        wandb.init(project="IK_model", name="flow-based ik")
    train_dataset = LiberoSuiteDataset4IK(
        suite_path = "/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero/libero_spatial",
        ratio=0.9,
        extra_state_keys=extra_state_keys,
        augmentation=True,
        mode="train",
        )
    valid_dataset = LiberoSuiteDataset4IK(
        suite_path = "/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero/libero_spatial",
        ratio=0.1,
        extra_state_keys=extra_state_keys,
        augmentation=False,
        mode="validation",
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True, num_workers=8)

    for epoch in range(args.epoch):
        with tqdm(total=len(train_loader), desc=f"Epoch: {epoch + 1}/{args.epoch}") as pbar:
            resnet.train()
            for batch in train_loader:
                obs_goal, _, _, _ = batch
                visual_obs = obs_goal["side_view"].to(device)
                if use_vae:
                    visual_obs = vae(visual_obs).sample
                
                ee_state = obs_goal['ee_states'].to(device)
                weight = torch.ones([1, 8], device=device)
                # weight[-1] = 10
                loss = resnet.train_loss(ee_state, visual_obs, loss_type="huber_loss", weight=weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.use_wandb:
                    wandb.log({"loss": loss.item(),
                               "epoch": epoch,
                               "lr": optimizer.param_groups[0]["lr"]})
                    
                pbar.update(1)

        lr_scheduler.step()
        if epoch % args.evaluation_every == 0:  
            with torch.no_grad():
                resnet.eval()
                valid_loss_list=[]
                for batch in valid_loader:
                    obs_goal, _, _, _ = batch
                    visual_obs = obs_goal["side_view"].to(device)
                    if use_vae:
                        visual_obs = vae(visual_obs).sample
                    ee_state = obs_goal["ee_states"].to(device)

                    valid_loss = resnet.train_loss(ee_state, visual_obs, loss_type="L2")
                    valid_loss_list.append(valid_loss.item())
                
                os.makedirs("results/pixel_ik", exist_ok=True)
                torch.save(resnet.state_dict(), f"results/pixel_ik/model_{epoch}.pt")
                if args.use_wandb:
                    wandb.log({"valid_loss": np.mean(valid_loss_list)})

def prepare_vae(device):
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae = PeftModel.from_pretrained(vae, "/mnt/home/ZhangXiaoxiong/Documents/AVDC/lora-vae").eval()
    vae.requires_grad_(False)
    return vae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", default = False, type = bool)
    parser.add_argument("--epoch", default = 100, type = int)
    parser.add_argument("--evaluation_every", default = 5, type = int)

    args = parser.parse_args()
    main(args)