from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from torchvision.transforms import transforms
from diffusion_model.UNets import UnetBridge as Unet
from diffusion_model.GoalDiffusion import GoalGaussianDiffusion
from ema_pytorch import EMA
import torchvision
import imageio
import numpy as np
import torch
image_size = (48, 64)
device = torch.device('cuda')
model_path = "/mnt/home/ZhangXiaoxiong/Documents/results/libero-close-loop/model-6.pt"
pretrained_model = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(device)
text_encoder.requires_grad_(False)
text_encoder.eval()
model = Unet().to(device)
diffusion = GoalGaussianDiffusion(
    channels=3*6,
    model=model,
    image_size=image_size,
    timesteps=100,
    sampling_timesteps=10,
    loss_type='l2',
    objective='pred_v',
    beta_schedule = 'cosine',
    min_snr_loss_weight = True,
).to(device)
diffusion = EMA(diffusion)
diffusion.load_state_dict(torch.load(model_path)['ema'])
diffusion = diffusion.ema_model

image_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])
task_prompt = ["pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"]
image = Image.open("/mnt/home/ZhangXiaoxiong/Documents/VideoGeneration/results/policy/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate/video3.gif")

text_ids = tokenizer(task_prompt, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(device)
text_embedding = text_encoder(**text_ids).last_hidden_state
video_list = []
for time in range(image.n_frames):
    image.seek(time)
    frame = image.copy().convert('RGB')
    x_cond = image_transform(frame).unsqueeze(0).to(device)
    output = diffusion.sample(x_cond, text_embedding, batch_size=1).detach().squeeze(0)
    output = output.reshape(-1, 3, *image_size)
    
    video_clip = (torch.cat([x_cond, output], dim=0).cpu().numpy().transpose(0, 2, 3, 1).clip(0,1)*255).astype('uint8')
    video_clip = [frame for frame in video_clip]
    video_list.extend(video_clip)

imageio.mimsave(f"./visualization4.gif", video_list, duration=20)

