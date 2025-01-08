from diffusers.models import AutoencoderKL
from PIL import Image 
from torchvision import transforms
import torchvision 
import torch
from peft import PeftConfig, PeftModel


config = PeftConfig.from_pretrained("/mnt/home/ZhangXiaoxiong/Documents/AVDC/lora-vae")
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
lora_model = PeftModel.from_pretrained(vae, "/mnt/home/ZhangXiaoxiong/Documents/AVDC/lora-vae")
image = Image.open("/mnt/home/ZhangXiaoxiong/Documents/AVDC/test.png")
transform = transforms.Compose([
    transforms.ToTensor()
])
image = transform(image)
latent = vae.tiled_encode(image.unsqueeze(0), return_dict=True).latent_dist.mean
sample = vae.tiled_decode(latent).sample

torchvision.utils.save_image(torch.cat([sample.squeeze(0), image.squeeze(0)],dim=-1), "/mnt/home/ZhangXiaoxiong/Documents/AVDC/test-vae.png")