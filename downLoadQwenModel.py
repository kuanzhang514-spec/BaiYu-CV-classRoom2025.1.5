'''
下载 Qwen3-VL-4B-Instruct 模型到本地指定目录
下载路径: D:\qwen3_deploy
'''

from transformers import AutoModelForCausalLM, AutoProcessor
import os

# 创建下载目录（如果不存在）
download_dir = "D:\\qwen3_deploy"
os.makedirs(download_dir, exist_ok=True)

model_id = "Qwen/Qwen3-VL-4B-Instruct"

print(f"开始下载模型 {model_id} 到 {download_dir} ...")
print("（首次运行需下载约 9GB 数据，请保持网络畅通）")

# 仅下载模型文件（不加载到内存）
AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=download_dir,
    trust_remote_code=True,
    resume_download=True,      # 支持断点续传
    local_files_only=False     # 允许从网络下载
)

# 下载 processor（tokenizer + image processor）
AutoProcessor.from_pretrained(
    model_id,
    cache_dir=download_dir,
    trust_remote_code=True,
    resume_download=True,
    local_files_only=False
)

print("✅ 模型和处理器已成功下载到:", download_dir)