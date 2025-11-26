from diffusion_model.UNets import UnetBridge as Unet
from diffusion_model.GoalDiffusion import GoalGaussianDiffusion
from policy.vilt import BCViLTPolicy
from diffusion_model.FeatureExtractor import FeatureExtractor

from transformers import CLIPTextModel, CLIPTokenizer
from omegaconf import OmegaConf
from ema_pytorch import EMA
import torch


def prepare_feature_extractor(model_path, device, image_size=(48, 64)):
    model = Unet()
    diffusion = GoalGaussianDiffusion(
        channels=3 * 6,
        model=model,
        image_size=image_size,
        timesteps=100,
        sampling_timesteps=10,
        loss_type="l2",
        objective="pred_v",
        beta_schedule="cosine",
        min_snr_loss_weight=True,
    )
    diffusion = EMA(diffusion)
    diffusion.load_state_dict(torch.load(model_path)["ema"])
    diffusion = diffusion.ema_model
    unet = Unet()  # the UnetBridge class
    unet.load_state_dict(diffusion.model.state_dict())
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    feature_extractor = (
        FeatureExtractor(
            unet=unet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            diffusion_image_size=image_size,
        )
        .to(device)
        .eval()
    )

    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor


def replace_checkpoint_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    return new_state_dict


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
