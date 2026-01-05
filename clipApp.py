'''
这个文件是下载CLIP模型的程序

首先：需要事先手动在https://huggingface.co/openai/clip-vit-base-patch32/tree/main
页面下载以下所有json文件并放入同一文件夹中：
1. `config.json`
2. `preprocessor_config.json`
3. `tokenizer_config.json`
4. `vocab.json`
5. `merges.txt`
6. `special_tokens_map.json`
'''

#1.使用这个代码下载pytorch_model.bin文件，
# 下载好后会报错再给你重新下载新的model.safetensors文件，请把这个文件与之前手动下载json文件放在同一个文件夹里
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
save_path = "D:\\qwen3_deploy"   #指定模型文件的下载路径
# 加载（会自动下载到 D 盘，下次运行直接从 D 盘读）
model = CLIPModel.from_pretrained(model_id, cache_dir=save_path).to(device)
processor = CLIPProcessor.from_pretrained(model_id, cache_dir=save_path)
print(f"CLIP 模型已就绪，运行在: {device}")


# 这是测试CLIP模型下载到本地了能不能启动的代码 local_path 是模型下载的路径，里面有json和model.safetensors
# 这是测试CLIP模型下载到本地了能不能启动的代码 local_path 是模型下载的路径，里面有json和model.safetensors
# 这是测试CLIP模型下载到本地了能不能启动的代码 local_path 是模型下载的路径，里面有json和model.safetensors
# import torch
# from transformers import CLIPProcessor, CLIPModel
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # local_path 直接指向包含 model.safetensors 和 config.json 的那一层文件夹
# # 请确保这个文件夹里也有 config.json 和 preprocessor_config.json
# local_path = "D:\\qwen3_deploy\\models--openai--clip-vit-base-patch32\\snapshots\\c237dc49a33fc61debc9276459120b7eac67e7ef"
# try:
#     # 直接从绝对路径加载，不再通过 cache_dir 机制
#     model = CLIPModel.from_pretrained(local_path).to(device)
#     processor = CLIPProcessor.from_pretrained(local_path)
#     print(f"成功从本地绝对路径加载模型！运行在: {device}")
# except Exception as e:
#     print(f"加载失败：{e}")
