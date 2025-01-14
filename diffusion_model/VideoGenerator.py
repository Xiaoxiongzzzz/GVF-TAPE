import torch 
from torch import nn
from einops import rearrange
from diffusion_model.UNets import UnetLatent, UnetMW
from diffusion_model.RectifiedFlow import RectifiedFlow
from diffusers.models import AutoencoderKL
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer


class LatentVideoGenerator(nn.Module):
    def __init__(self, model, vae, tokenizer, text_encoder, rectified_flow, device):
        super().__init__()
        self.device = device
        self.model = model
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.rectified_flow = rectified_flow

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
        x_cond = x_cond/255.
        task_embed = self.encode_batch_text(batch_text)
        x_cond_encode = self.vae.encode(x_cond).latent_dist.mean.mul_(self.vae.config.scaling_factor)
        h, w = x_cond_encode.shape[2], x_cond_encode.shape[3]

        noise = torch.randn((B, (4*self.f), h, w), device = self.device)
        x = self.rectified_flow.sample(self.model, noise, x_cond_encode, task_embed)
        x = rearrange(x, "b (f c) h w -> (b f) c h w", f=self.f)
        x = self.vae.decode(x.mul_(1/self.vae.config.scaling_factor)).sample
        x = rearrange(x, "(b f) c h w -> b (f c) h w", f=self.f)

        return x
    def encode_batch_text(self, batch_text):
        
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state

        return batch_text_embed

class PixelVideoGenerator(nn.Module):
    def __init__(self, model, tokenizer, text_encoder, rectified_flow, device):
        super().__init__()
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.rectified_flow = rectified_flow

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
        x_cond = x_cond/255.
        task_embed = self.encode_batch_text(batch_text)
        # task_embed = task_embed.repeat_interleave(10, dim=0)
        h, w = x_cond.shape[2], x_cond.shape[3]

        noise = torch.randn((B, (3*self.f), h, w), device = self.device)
        x = self.rectified_flow.sample(self.model, noise, x_cond, task_embed)

        return x
    def encode_batch_text(self, batch_text):
        
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state

        return batch_text_embed
def prepare_video_generator(unet_path, device, sample_timestep=10, latent=False):
    unet = UnetLatent().to(device) if latent else UnetMW().to(device) 
    unet.load_state_dict(torch.load(unet_path)["model"])
    unet.eval().requires_grad_(False)

    # Language Model
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    rectified_flow = RectifiedFlow(sample_timestep=sample_timestep)

    if latent:
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device).eval()
        vae.requires_grad_(False)
        video_generator = LatentVideoGenerator(unet, vae, tokenizer, text_encoder, rectified_flow, device)
    else:
        video_generator = PixelVideoGenerator(unet, tokenizer, text_encoder, rectified_flow, device)

    return video_generator

# Test Code
# if __name__ == "__main__":
#     text = "open the drawer"
#     x_cond = torch.randn(1, 3, 480, 640).to(torch.device("cuda"))
#     video_generator = prepare_video_generator("/mnt/home/ZhangXiaoxiong/Documents/AVDC/results/RFlow_real_world/model_99000.pt", 
#                                                 device=torch.device("cuda"),
#                                                 latent=True)
#     x = video_generator(text, x_cond)
#     import ipdb; ipdb.set_trace()