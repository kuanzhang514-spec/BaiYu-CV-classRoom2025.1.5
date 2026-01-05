'''
评估：
Qwen 原始回答的准确率（qwen_original vs gt_answers）
CLIP 增强后最终答案的准确率（final_answer vs gt_answers）
'''


import json
import re

def normalize_text(text):
    """标准化文本：转小写、去空格、去标点（可选）"""
    if not isinstance(text, str):
        return ""
    # 保留字母、数字、空格，其余替换为空格
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    return ' '.join(text.split())  # 合并多余空格

def is_correct(pred, gt_list):
    """
    判断预测是否正确（宽松匹配）
    - pred: 字符串
    - gt_list: 标准答案列表
    返回 True/False
    """
    if not pred:
        return False
    
    pred_norm = normalize_text(pred)
    valid_gt = []
    for ans in gt_list:
        if isinstance(ans, str) and ans.strip() and ans.lower() not in ["unanswerable", "none", "unknown"]:
            valid_gt.append(normalize_text(ans))
    
    if not valid_gt:
        return False

    # 方法：检查 pred 是否包含任一 gt，或任一 gt 是否包含 pred（双向子串）
    for gt in valid_gt:
        if gt in pred_norm or pred_norm in gt:
            return True
    return False

def main(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    qwen_correct = 0
    final_correct = 0

    for item in data:
        gt_answers = item.get("gt_answers", [])
        
        # Qwen 原始回答
        qwen_ans = item.get("qwen_original", "").strip()
        if is_correct(qwen_ans, gt_answers):
            qwen_correct += 1

        # CLIP 增强后的最终回答
        final_ans = item.get("final_answer", "").strip()
        if is_correct(final_ans, gt_answers):
            final_correct += 1

    qwen_acc = qwen_correct / total * 100
    final_acc = final_correct / total * 100

    print(f"总样本数: {total}")
    print(f"Qwen 原始回答准确率: {qwen_correct}/{total} = {qwen_acc:.2f}%")
    print(f"CLIP 增强后准确率: {final_correct}/{total} = {final_acc:.2f}%")
    print(f"提升: {final_acc - qwen_acc:.2f} 个百分点")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="D:\qwen3_deploy\\results\\vqa_results.json", help="VQA 结果 JSON 文件路径")
    args = parser.parse_args()
    main(args.json)