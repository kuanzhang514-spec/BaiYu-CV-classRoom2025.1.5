'''
这个文件是测试一下CLIP模型能不能推理的程序

输入一张有两只猫的图片，
推理结果:
a photo of a cat: 0.9949
a photo of a dog: 0.0051
'''

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import sys
import os

# 模型本地路径
local_path = "D:\\qwen3_deploy\\models--openai--clip-vit-base-patch32\\snapshots\\c237dc49a33fc61debc9276459120b7eac67e7ef"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("正在加载 CLIP 模型...")
model = CLIPModel.from_pretrained(local_path).to(device)
processor = CLIPProcessor.from_pretrained(local_path)
print("模型加载完成！")

# 从终端获取图片路径
img_path = input("请输入图片路径（支持本地路径或 HTTP/HTTPS 链接）: ").strip()

if not img_path:
    print("错误：未提供图片路径。")
    sys.exit(1)

# 加载图片：支持本地文件或网络 URL
try:
    if img_path.startswith(("http://", "https://")):
        image = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
    else:
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"本地文件不存在: {img_path}")
        image = Image.open(img_path).convert("RGB")
except Exception as e:
    print(f"加载图片失败: {e}")
    sys.exit(1)

# 预设文本标签（你也可以扩展或改为输入）
text_labels = ["a photo of a cat", "a photo of a dog"]

# 处理输入
inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True).to(device)

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图像-文本相似度
    probs = logits_per_image.softmax(dim=1)      # 转为概率分布

# 输出结果
print("\n推理结果:")
for i, label in enumerate(text_labels):
    print(f"{label}: {probs[0][i].item():.4f}")

