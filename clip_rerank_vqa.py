'''
CLIP + Qwen ： 这是用CLIP增强Qwen 的脚本

1.流程：
使用 Qwen3-VL 模型生成 top-k 候选答案及其概率
使用 CLIP 模型计算每个候选答案与图像的语义相似度
融合 Qwen 的概率 + CLIP 的相似度，加权得到最终得分
选择得分最高的答案作为最终预测
评估准确率并保存结果（JSON + HTML 可视化）

2.数据集json文件metadata.json看懂：
answers：是 VQA 数据集的标准答案列表，每个问题通常由多个标注者独立作答，因此是一个包含多个字符串的列表

3.结果生成的json文件看懂：
Ground Truth (GT)：是人工标注的标准答案列表，系统判断时会忽略无效项（如非字符串、"unanswerable"），只关注有效答案。
Qwen Original（原始预测）：是 Qwen3-VL 模型在 beam search 中排名第一（最高概率）的生成结果
Final Answer（最终答案）：在所有候选中，第一个候选的融合得分最高，就被选为最终答案
qwen_prob：Qwen 认为该序列的整体生成概率（经 softmax 归一化）,这个数字是在 Qwen 生成的几个答案里，这个答案排第几、有多被模型喜欢
clip_sim：CLIP 计算的图像-文本匹配分数（值越大越相关）
final_score：按公式 0.8 * norm_qwen + 0.2 * norm_clip 融合后的得分

'''
import os
import json
import re
import torch
from PIL import Image
from tqdm import tqdm
import argparse

# Qwen3-VL 模型封装
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

class Qwen3VLCandidateGenerator:
    def __init__(self, model_path):
        print("正在加载 Qwen3-VL 量化模型...")
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
        print("Qwen3-VL 加载成功！")

    def generate_candidates(self, image_path, question, num_beams=5, max_new_tokens=50):
        """
        返回 top-k 候选答案（强制简洁，仅核心内容）
        """
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((384, 384))

        # 提示词： 仅回答关键信息，需要很精简
        concise_question = f"{question} Only answer key information, it needs to be very concise."

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
            
            # 再进一步清理回答：只保留第一句，并去除常见引导语（去掉可能残留的句首模板）
            # 例如：如果回答是 "The brand is Dakota Digital." → 提取 "Dakota Digital"
            if answer.lower().startswith(("the answer is", "answer:", "it is", "the brand is", "the number is")):
                # 用正则提取冒号或 is 后的内容
                import re
                match = re.search(r'(?:is|:)\s*(.+)', answer, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip().rstrip('.,')
            
            # 确保非空
            if not answer:
                answer = "unknown"
                
            candidates.append({
                "text": answer,
                "qwen_prob": probs[i]
            })
        return candidates


# CLIP 模型封装
from transformers import CLIPProcessor, CLIPModel

def min_max_norm(scores):
    if len(scores) == 1:
        return [1.0]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).strip()

#一些噪声（如 "la grande "）会被该函数过滤掉
def is_correct(pred, gt_list):
    '''
    举个例子：
    清洗预测答案 → 提取关键词 "dakota digital"
    清洗 GT → 得到 ["dakota", "dakota digital", ...]
    检查："dakota" 或 "dakota digital" 是否出现在预测文本中？
    预测文本包含 “Dakota Digital” →  匹配成功！
    '''
    pred_clean = clean_text(pred)
    if not pred_clean:
        return False
    for gt in gt_list:
        if not isinstance(gt, str):
            continue
        gt_clean = clean_text(gt)
        if not gt_clean or gt_clean in ["unanswerable", "none"]:
            continue
        if gt_clean in pred_clean:
            return True
    return False

class CLIPReranker:
    '''
    CLIP 的工作机制：
    输入：一张图像 + 一组任意文本描述（比如你的候选答案）
    输出：每个文本与图像的 相似度分数（logits）
    '''
    def __init__(self, model_path):
        print("正在加载 CLIP 模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        print("CLIP 加载成功！")
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


# 主流程
def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化模型
    qwen_gen = Qwen3VLCandidateGenerator(args.qwen_model_path)
    clip_reranker = CLIPReranker(args.clip_model_path)

    # 加载数据集
    with open(args.metadata_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    correct = 0
    total = 0

    for item in tqdm(data[:args.max_samples], desc="Processing VQA samples"):
        img_path = os.path.join(args.image_root, item["image_file"])
        if not os.path.exists(img_path):
            continue

        question = item["question"]
        gt_answers = [ans.lower().strip() for ans in item["answers"] if isinstance(ans, str)]
        if not gt_answers:
            continue

        try:
            candidates = qwen_gen.generate_candidates(img_path, question, num_beams=args.num_beams)
        except Exception as e:
            print(f"Qwen failed on {img_path}: {e}")
            continue

        candidate_texts = [cand["text"] for cand in candidates]

        try:
            clip_sims = clip_reranker.compute_similarity(img_path, candidate_texts)
        except Exception as e:
            print(f"CLIP failed on {img_path}: {e}")
            continue

        qwen_probs = [cand["qwen_prob"] for cand in candidates]
        norm_qwen = min_max_norm(qwen_probs)
        norm_clip = min_max_norm(clip_sims)

        final_scores = [
            args.w1 * norm_qwen[i] + args.w2 * norm_clip[i]
            for i in range(len(candidates))
        ]

        best_idx = final_scores.index(max(final_scores))
        final_answer = candidates[best_idx]["text"]

        is_correct_flag = is_correct(final_answer, gt_answers)
        if is_correct_flag:
            correct += 1
        total += 1

        results.append({
            "image_file": item["image_file"],
            "question": question,
            "gt_answers": gt_answers,
            "qwen_original": candidates[0]["text"],
            "final_answer": final_answer,
            "candidates": [
                {
                    "text": cand["text"],
                    "qwen_prob": round(cand["qwen_prob"], 4),
                    "clip_sim": round(clip_sims[i], 4),
                    "final_score": round(final_scores[i], 4)
                }
                for i, cand in enumerate(candidates)
            ],
            "correct": is_correct_flag
        })

    acc = correct / total if total > 0 else 0
    print(f"\n Final Accuracy: {acc:.4f} ({correct}/{total})")

    with open(os.path.join(args.output_dir, "vqa_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    html_content = "<html><body><h2>CLIP-Reranked VQA Results</h2>"
    for res in results:
        html_content += f"""
        <div style="border:1px solid #ccc; margin:10px; padding:10px;">
            <img src="{os.path.join(args.image_root, res['image_file'])}" width="200">
            <p><b>Q:</b> {res['question']}</p>
            <p><b>GT:</b> {', '.join(res['gt_answers'])}</p>
            <p><b>Qwen Original:</b> {res['qwen_original']}</p>
            <p><b>Final Answer:</b> <span style="color:{'green' if res['correct'] else 'red'}">{res['final_answer']}</span></p>
            <details><summary>Candidates</summary><pre>{json.dumps(res['candidates'], indent=2)}</pre></details>
        </div>
        """
    html_content += "</body></html>"
    with open(os.path.join(args.output_dir, "visualization.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_model_path", type=str, default="D:/qwen3_deploy/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17")
    parser.add_argument("--clip_model_path", type=str, default="D:\\qwen3_deploy\\models--openai--clip-vit-base-patch32\\snapshots\\c237dc49a33fc61debc9276459120b7eac67e7ef")
    parser.add_argument("--metadata_json", type=str, default="./images/metadata.json")
    parser.add_argument("--image_root", type=str, default="\\qwen3_deploy\\images")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--w1", type=float, default=0.8)
    parser.add_argument("--w2", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=50)

    args = parser.parse_args()
    main(args)
