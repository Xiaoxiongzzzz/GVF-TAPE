from torchvision import transforms
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from peft import PeftModel
import h5py
import torch
import cv2
device = torch.device("cuda")
to_tensor = transforms.ToTensor()
gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2))

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
vae = PeftModel.from_pretrained(vae, "/mnt/home/ZhangXiaoxiong/Documents/AVDC/lora-vae").eval()
vae.requires_grad_(False)

file_path = "/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
with h5py.File(file_path) as f:
    origin_image = f["data"]["demo_0"]["obs"]["agentview_rgb"][50]
    origin_image = cv2.flip(origin_image, 0)
    origin_image = to_tensor(origin_image)
    
    # latent = vae.encode(origin_image.unsqueeze(0).to(device)).latent_dist.mean
    # process_image = vae.decode(latent).sample
    process_image = vae(origin_image.unsqueeze(0).to(device)).sample
import ipdb; ipdb.set_trace()
concat_image = torch.cat([origin_image, process_image.cpu().squeeze(0)], dim=2)
save_image(concat_image, "./blur.png")

