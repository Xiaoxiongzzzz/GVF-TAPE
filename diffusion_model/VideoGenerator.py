import torch
from torch import nn
from einops import rearrange
from diffusion_model.UNets import UnetLatent, UnetMW
from diffusion_model.RectifiedFlow import RectifiedFlow

# from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from diffusion_model.GoalDiffusion import GoalGaussianDiffusion
from ema_pytorch import EMA
from torchvision import transforms


class LatentVideoGenerator(nn.Module):
    def __init__(
        self,
        model,
        vae,
        tokenizer,
        text_encoder,
        rectified_flow,
        device,
        size=(512, 512),
    ):
        super().__init__()
        self.device = device
        self.model = model
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.rectified_flow = rectified_flow
        self.resize = transforms.Resize(size)

        self.f = 6

    def forward(self, batch_text, x_cond: torch.Tensor):
        """
        Input:
            batch_text: str or list of str
            x_cond: shape = [b, c, h, w]
        Output:
            x: shape = [b, (f, c), h, w]
        """
        B = x_cond.shape[0]
        x_cond = self.resize(x_cond)
        x_cond = x_cond / 255.0
        task_embed = self.encode_batch_text(batch_text)
        x_cond_encode = self.vae.encode(x_cond).latent_dist.mean.mul_(
            self.vae.config.scaling_factor
        )
        h, w = x_cond_encode.shape[2], x_cond_encode.shape[3]

        noise = torch.randn((B, (4 * self.f), h, w), device=self.device)
        x = self.rectified_flow.sample(self.model, noise, x_cond_encode, task_embed)
        x = rearrange(x, "b (f c) h w -> (b f) c h w", f=self.f)
        x = self.vae.decode(x.mul_(1 / self.vae.config.scaling_factor)).sample
        x = rearrange(x, "(b f) c h w -> b (f c) h w", f=self.f)

        return x

    def encode_batch_text(self, batch_text):

        batch_text_ids = self.tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state

        return batch_text_embed


class PixelVideoGenerator(nn.Module):
    def __init__(
        self, model, tokenizer, text_encoder, rectified_flow, device, depth=False
    ):
        super().__init__()
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.rectified_flow = rectified_flow
        self.depth = depth
        self.channel = 3 if not depth else 4
        self.f = 6

    def forward(self, batch_text_or_embed, x_cond: torch.Tensor):
        """
        Input:
            batch_text_or_embed: str/list of str, or tensor of shape [b, n, embed_dim]
            x_cond: shape = [b, c(3), h, w]
        Output:
            x: shape = [b, (f, c), h, w]
        """
        B = x_cond.shape[0]
        x_cond = x_cond / 255.0

        # Handle different types of input
        if isinstance(batch_text_or_embed, (str, list)):
            # If input is text, encode it
            task_embed = self.encode_batch_text(batch_text_or_embed)
        else:
            # If input is already encoded embedding
            task_embed = batch_text_or_embed
            if len(task_embed.shape) == 3:
                # Ensure the embedding is on the correct device
                task_embed = task_embed.to(self.device)
            else:
                raise ValueError(
                    "Text embedding should be a 3D tensor of shape [batch_size, n, embed_dim]"
                )

        h, w = x_cond.shape[2], x_cond.shape[3]
        noise = torch.randn((B, (self.channel * self.f), h, w), device=self.device)
        x = self.rectified_flow.sample(self.model, noise, x_cond, task_embed)

        return x

    def encode_batch_text(self, batch_text):

        batch_text_ids = self.tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state

        return batch_text_embed


def prepare_video_generator(
    unet_path, device, sample_timestep=10, latent=False, depth=False
):
    print(f"Using GPU: {device}")
    # torch.cuda.empty_cache()
    unet = UnetLatent().to(device) if latent else UnetMW(depth).to(device)
    # torch.cuda.empty_cache()
    unet.load_state_dict(torch.load(unet_path)["model"])
    # torch.cuda.empty_cache()
    unet.eval().requires_grad_(False)

    # Language Model
    # pretrained_model = "openai/clip-vit-base-patch32"
    # tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    # text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(device)
    # text_encoder.requires_grad_(False)
    # text_encoder.eval()
    tokenizer = None
    text_encoder = None

    rectified_flow = RectifiedFlow(sample_timestep=sample_timestep)

    if latent:
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device).eval()
        vae.requires_grad_(False)
        video_generator = LatentVideoGenerator(
            unet, vae, tokenizer, text_encoder, rectified_flow, device
        )
    else:
        video_generator = PixelVideoGenerator(
            unet, tokenizer, text_encoder, rectified_flow, device, depth
        )

    return video_generator


class DiffusionVideoGenerator(nn.Module):
    def __init__(self, diffusion_model, tokenizer, text_encoder, device):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device

        self.f = 6

    def forward(self, batch_text, x_cond):
        """
        Input:
            batch_text: str or list of str
            x_cond: shape = [b, c, h, w]
        Output:
            x: shape = [b, (f, c), h, w]
        """
        B = x_cond.shape[0]
        x_cond = x_cond / 255.0
        task_embed = self.encode_batch_text(batch_text)
        x = self.diffusion_model.sample(x_cond, task_embed, batch_size=B)

        return x

    def encode_batch_text(self, batch_text):

        batch_text_ids = self.tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state

        return batch_text_embed


def prepare_diffusion_video_generator(
    diffusion_model_path, device, sample_timestep=10, image_size=(128, 128)
):
    diffusion_model = EMA(
        GoalGaussianDiffusion(
            channels=3 * (7 - 1),
            model=UnetMW(),
            image_size=image_size,
            timesteps=100,
            sampling_timesteps=sample_timestep,
            loss_type="l2",
            objective="pred_v",
            beta_schedule="cosine",
            min_snr_loss_weight=True,
        )
    )
    diffusion_model.load_state_dict(torch.load(diffusion_model_path)["ema"])
    diffusion_model.eval().to(device)

    # Language Model
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion_video_generator = DiffusionVideoGenerator(
        diffusion_model.ema_model, tokenizer, text_encoder, device
    )

    return diffusion_video_generator


# # Test Code
# if __name__ == "__main__":
#     text = "open the drawer"
#     x_cond = torch.randn(1, 3, 128, 128).to(torch.device("cuda"))
#     video_generator = prepare_video_generator("/mnt/data0/xiaoxiong/single_view_goal_diffusion/results/RFlow_libero_goal_depth/ckpt/model_0.pt",
#                                                 device=torch.device("cuda"),
#                                                 sample_timestep=10,
#                                                 depth=True)
#     sample = video_generator(text, x_cond)
#     import ipdb; ipdb.set_trace()
