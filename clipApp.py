'''
这个文件是下载CLIP模型的程序(非第一版)
手动指定下载到 local_path
'''
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# 直接指向包含 model.safetensors 和 config.json 的那一层文件夹
# 请确保这个文件夹里也有 config.json 和 preprocessor_config.json
local_path = "D:\\qwen3_deploy\\models--openai--clip-vit-base-patch32\\snapshots\\c237dc49a33fc61debc9276459120b7eac67e7ef"

try:
    # 直接从绝对路径加载，不再通过 cache_dir 机制
    model = CLIPModel.from_pretrained(local_path).to(device)
    processor = CLIPProcessor.from_pretrained(local_path)
    
    print(f"成功从本地绝对路径加载模型！运行在: {device}")
except Exception as e:
    print(f"加载失败：{e}")
