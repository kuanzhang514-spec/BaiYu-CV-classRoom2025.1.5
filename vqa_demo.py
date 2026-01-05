import os
import re
import torch
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor
)

# ----------------------------
# Qwen3-VL 候选生成器
# ----------------------------
class Qwen3VLCandidateGenerator:
    def __init__(self, model_path):
        print("正在加载 Qwen3-VL 模型（4-bit 量化）...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        max_memory = {0: "3.5GiB", "cpu": "16GiB"}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            offload_folder="D:/qwen3_deploy/offload"
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("✅ Qwen3-VL 加载成功！")

    def generate_candidates(self, image_path, question, num_beams=3, max_new_tokens=20):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((384, 384))
        
        concise_question = f"{question} Answer with only the key information, no explanation."
        
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": concise_question}]
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[1]
        sequences = out.sequences
        scores = out.sequences_scores

        candidates = []
        probs = torch.softmax(scores, dim=0).tolist()
        for i in range(num_beams):
            gen_ids = sequences[i][input_len:]
            answer = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
            
            if answer.lower().startswith(("the answer is", "answer:", "it is", "the brand is", "the number is")):
                match = re.search(r'(?:is|:)\s*(.+)', answer, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip().rstrip('.,')
            if not answer:
                answer = "unknown"
                
            candidates.append({"text": answer, "qwen_prob": probs[i]})
        return candidates

# ----------------------------
# CLIP 重排序器
# ----------------------------
def min_max_norm(scores):
    if len(scores) == 1:
        return [1.0]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

class CLIPReranker:
    def __init__(self, model_path):
        print("正在加载 CLIP 模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        print("✅ CLIP 加载成功！")

    def compute_similarity(self, image_path, candidate_texts):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=candidate_texts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image.squeeze(0).tolist()
        return logits_per_image

# ----------------------------
# 主函数：交互式输入
# ----------------------------
def main():
    # ====== 配置你的模型路径 ======
    QWEN_MODEL_PATH = "D:/qwen3_deploy/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
    CLIP_MODEL_PATH = "D:\\qwen3_deploy\\models--openai--clip-vit-base-patch32\\snapshots\\c237dc49a33fc61debc9276459120b7eac67e7ef"
    
    # 可调参数
    NUM_BEAMS = 3
    MAX_NEW_TOKENS = 50
    W1, W2 = 0.8, 0.2  # Qwen 权重, CLIP 权重

    print("🚀 欢迎使用 VQA 交互式 Demo！")
    print("请输入以下信息：\n")

    # 1. 输入图片路径
    while True:
        image_path = input("📁 请输入图片路径（例如: ./images/0.jpg）: ").strip().strip('"\'')
        if os.path.exists(image_path):
            break
        else:
            print(f"❌ 图片不存在，请重新输入。\n")

    # 2. 输入问题
    question = input("❓ 请输入问题（例如: what is the brand of this camera?）: ").strip()
    if not question:
        question = "What is in the image?"

    # 3. 加载模型（首次运行时加载）
    print("\n⏳ 正在初始化模型（首次加载较慢）...")
    qwen_gen = Qwen3VLCandidateGenerator(QWEN_MODEL_PATH)
    clip_reranker = CLIPReranker(CLIP_MODEL_PATH)

    # 4. 推理
    print("\n🔍 正在生成候选答案...")
    candidates = qwen_gen.generate_candidates(
        image_path, 
        question, 
        num_beams=NUM_BEAMS,
        max_new_tokens=MAX_NEW_TOKENS
    )
    candidate_texts = [cand["text"] for cand in candidates]

    print("🔄 正在用 CLIP 重排序...")
    clip_sims = clip_reranker.compute_similarity(image_path, candidate_texts)

    # 5. 融合打分
    qwen_probs = [cand["qwen_prob"] for cand in candidates]
    norm_qwen = min_max_norm(qwen_probs)
    norm_clip = min_max_norm(clip_sims)
    final_scores = [
        W1 * norm_qwen[i] + W2 * norm_clip[i]
        for i in range(len(candidates))
    ]

    best_idx = final_scores.index(max(final_scores))
    final_answer = candidates[best_idx]["text"]

    # 6. 输出结果
    print("\n" + "="*60)
    print("✅ 推理完成！")
    print(f"📸 图片: {image_path}")
    print(f"❓ 问题: {question}")
    print(f"\n🟢 Qwen 最高概率回答: {candidates[0]['text']}")
    print(f"🟣 CLIP 增强后最终答案: {final_answer}")

    print("\n📋 候选答案详情:")
    for i, cand in enumerate(candidates):
        mark = " ← 最佳" if i == best_idx else ""
        print(f"  {i+1}. \"{cand['text']}\"")
        print(f"      Qwen Prob: {cand['qwen_prob']:.4f}")
        print(f"      CLIP Sim:  {clip_sims[i]:.4f}")
        print(f"      Final Score: {final_scores[i]:.4f}{mark}")
    print("="*60)

if __name__ == "__main__":
    main()