'''
加载已经下载好的Qwen模型，并量化处理
单模态的：输入文字问题，输出答案 
'''

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image

# 量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# 指向你本地模型的实际位置（快照目录）
LOCAL_MODEL_PATH = "D:/qwen3_deploy/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"

print("正在加载本地模型...")

# 加载模型
max_memory = {0: "3.5GiB", "cpu": "16GiB"}  # 假设你有 16GB 内存
model = Qwen3VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    max_memory=max_memory,
    trust_remote_code=True,
    dtype=torch.float16,
    offload_folder="D:/qwen3_deploy/offload"
)

# 加载处理器
processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True
)
print("模型加载成功！")

# 控制台获取输入的问题
text_request = input("请输入你的问题: ").strip()
messages = [{"role": "user", "content": [
    {"type": "text", "text": text_request}   #问题在这里
]}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 确保输入在正确的设备上

print("正在生成回答...")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True
    )
    input_ids = inputs["input_ids"]
    generated_ids = [o[len(i):] for i, o in zip(input_ids, out)]
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("\n模型回答：", result)
    