import torch
import imageio
import numpy as np
import torch.nn.functional as F 
from PIL import Image
from einops import repeat, rearrange
from torch import nn
from torchvision import transforms
from diffusion_model.model.nn import timestep_embedding
class FeatureExtractor(nn.Module):
    def __init__(self, unet: nn.Module,     #the UnetBridge class
                  tokenizer, 
                  text_encoder, 
                  device,
                  diffusion_image_size) -> None:
        super().__init__()

        self.unet = unet.unet            #get the original Unet class 
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device

        self.resize_diffusion_image = transforms.Resize(diffusion_image_size)

    def forward(self, x, task_prompt, t=0):
        '''
        Args:
            x: (B, time, channel, height, width)
            task_prompt: str or list of str
            t: int
        Return:
            pred: (b, time, channe, height, width)    
        '''
        f = 6
        B, T = x.shape[:2]
        spatial_size = x.shape[-2:]

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.preprocess_image(x)
        x = repeat(x, '(b t) c h w -> (b t) c f h w ', f=f, b=B, t=T)
        
        if isinstance(task_prompt, str):
            task_prompt = [task_prompt]
        text_ids = self.tokenizer(task_prompt, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        text_embedding = self.text_encoder(**text_ids).last_hidden_state
        text_embedding = text_embedding.unsqueeze(1).expand(-1, T, -1, -1)
        text_embedding = rearrange(text_embedding, "b t n c -> (b t) n c")

        noise  = torch.randn(B*T, 3, f, *spatial_size).to(self.device)
        x = torch.cat([noise, x], dim=1)

        t = t*torch.ones(B*T,).to(self.device).to(torch.int64)

        feature = self.extract_feature(x, t, text_embedding)
        feature = rearrange(feature, "(b t) c f h w -> b t c f h w", b=B, t=T)
        return feature
    def extract_feature(self, x, t, text_embedding):
        '''
        Args:
            x: (B*T 6 f h w)
            t: (B*T)
        '''
        indexs = [3, 7, 11]
        space_size = x.shape[-2:]
        emb = self.unet.time_embed(timestep_embedding(t, self.unet.model_channels))
        if self.unet.task_tokens:
            label_emb = self.unet.task_attnpool(text_embedding).mean(dim=1)
            emb = emb + label_emb
        
        hs = []
        h = x.type(self.unet.dtype)
        for module in self.unet.input_blocks:
            h = module(h, emb)
            hs.append(h)

        feature = [hs[i] for i in indexs]
        feature = [F.interpolate(f, size=(6,*space_size), mode='trilinear', align_corners=True) for f in feature]
        feature = torch.cat(feature, dim=1)
        return feature
    def preprocess_image(self, x):
        '''
            Resize input image to the needed image size and normalize image from 0-255 to 0-1
            x: (B, channel, height, width)
        '''
        x = self.resize_diffusion_image(x)
        return x/255.0
    def extract_from_list(self, xs: list, task_prompt):
        """
        Input:
            x: [(112, 112, 3)]
        """
        xs = [self.resize_diffusion_image(torch.from_numpy(x).to(self.device).permute(2, 0, 1)).unsqueeze(0) for x in xs]        #(b, c, h, w)
        x_cond = xs[0]
        x = torch.stack(xs[1:], dim=2)  #(b, 3, f, h, w)
        f = x.shape[2]
        x_cond = repeat(x_cond, "b c h w  -> b c f h w", f=f)
        x = torch.cat([x, x_cond], dim = 1)
        if isinstance(task_prompt, str):
            task_prompt = [task_prompt]
        text_ids = self.tokenizer(task_prompt, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        text_embedding = self.text_encoder(**text_ids).last_hidden_state
        t = 0*torch.ones(1,).to(self.device).to(torch.int64)

        return self.extract_feature(x, t, text_embedding)