import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
local_dir = "./weights"

snapshot_download(
    repo_id="Ackermannnnnn/CoRL_2025_GVF-TAPE",
    local_dir = local_dir,
    local_dir_use_symlinks=False,
)