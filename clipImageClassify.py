'''
CLIP：输入图片和分类标签，输出标签匹配度
'''

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

class CLIPImageClassifier:
    def __init__(self, model_path="D:\\qwen3_deploy\\models--openai--clip-vit-base-patch32\\snapshots\\c237dc49a33fc61debc9276459120b7eac67e7ef"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("正在加载 CLIP 模型...")
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        print("模型加载完成！")

    def classify_image(self, img_path, text_labels):
        try:
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"本地文件不存在: {img_path}")
            image = Image.open(img_path).convert("RGB")

            inputs = self.processor(text=text_labels, images=image, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # 图像-文本相似度
                probs = logits_per_image.softmax(dim=1)      # 转为概率分布
            
            # 输出结果
            print("\n推理结果:")
            for i, label in enumerate(text_labels):
                print(f"{label}: {probs[0][i].item():.4f}")
                
        except Exception as e:
            print(f"处理图片时发生错误: {e}")

if __name__ == "__main__":
    classifier = CLIPImageClassifier()
    while True:
        img_path = input("请输入图片路径（输入exit退出）: ").strip()
        if img_path.lower() == 'exit':
            break
        
        # 获取用户输入的文本标签，以逗号分隔
        text_labels_input = input("请输入您想要比较的文本标签（以逗号分隔）: ").strip()
        text_labels = [text.strip() for text in text_labels_input.split(",")]
        
        classifier.classify_image(img_path, text_labels)