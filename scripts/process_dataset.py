from transformers import CLIPTokenizer, CLIPTextModel
from glob import glob
from tqdm import tqdm
import torch
import os
import h5py
device = torch.device("cuda")
pretrained_model = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(device)
text_encoder.requires_grad_(False)
text_encoder.eval()

folder_path = "/mnt/home/ZhangXiaoxiong/Data/atm_data/atm_libero/libero_spatial"
file_path_list = glob(os.path.join(folder_path, "*.hdf5"), recursive=True)
for file_path in tqdm(file_path_list):
    text_prompt = os.path.basename(file_path).split(".")[0]
    text_prompt = text_prompt.replace("_", " ")[:-5]
    print(f"Processing {text_prompt}")
    with h5py.File(file_path, "r+") as f:
        data = f["data"]
        text_inputs = tokenizer(
            text_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_embed = text_encoder(**text_inputs)["pooler_output"]
        
        for demo_name in data.keys():
            demo = data[demo_name]
            if "task_embed" in demo:
                del demo["task_embed"]
            demo["task_embed"] = text_embed.detach().cpu().numpy()
