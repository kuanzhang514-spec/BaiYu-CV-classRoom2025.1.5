'''
加载已经下载好的Qwen模型，并量化处理
多模态的模型输入：输入图片和提问， 输出回答
这是不关闭模型的版本，可以连续提问
<< 最终使用的代码版本 >>
'''

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
def QWen3(img):
    # 量化配置
    LOCAL_MODEL_PATH = "D:/qwen3_deploy/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    print("正在加载本地模型...")

    max_memory = {0: "3.5GiB", "cpu": "16GiB"}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        LOCAL_MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        dtype=torch.float16,
        offload_folder="D:/qwen3_deploy/offload"
    )
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

    print("模型加载成功！")


    while True:
        img_path = img
        if img_path.lower() == 'exit':
            break

        try:
            image = Image.open(img_path).convert("RGB")
            image.thumbnail((384, 384))

            text_request = "1.请给我图片中前景物体的左上角和右下角坐标，模板格式是左上角坐标(x1,y1)，右下角坐标 (x2,y2)    2.请给我图片背景的左上角和右下角坐标，模板格式是左上角坐标(x3,y3)，右下角坐标(x4,y4)"
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_request}
            ]}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            print("正在生成回答中ing...")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True
                )
                input_ids = inputs["input_ids"]
                generated_ids = [o[len(i):] for i, o in zip(input_ids, out)]
                result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print("\nA-------模型回答：", result)
                return result
        except Exception as e:
            print(f"发生错误: {e}")
        # 每次处理完一个请求后不需要特别清理上下文，因为这里的上下文是基于单个请求构建的
    