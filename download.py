import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
# model_dir = snapshot_download('X-D-Lab/MindChat-Qwen-7B', cache_dir='./', revision='master')

model_dir = snapshot_download('X-D-Lab/MindChat-Qwen2-4B', cache_dir='./', revision='master')

