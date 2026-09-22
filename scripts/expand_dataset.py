"""数据扩增：用 LLM（DeepSeek API）把种子数据扩成大规模领域数据集。

用法：
  1. 设置环境变量 DEEPSEEK_API_KEY（或填入下方）
  2. python expand_dataset.py --input ../data/hrp_qa_seed.jsonl --output ../data/hrp_qa_expanded.jsonl --target 500

原理：把已有问答作为"范例"，让 LLM 基于同样的主题与风格生成新的 (instruction, output) 对。
生成的答案需要人工抽查；建议保留原种子数据作为质量锚点。
"""
import argparse
import json
import os
import random
import re
import time

def call_llm(api_key: str, examples: list[dict], n: int, topic: str) -> list[dict]:
    import httpx
    payload = {
        "model": "deepseek-chat",
        "temperature": 0.8,
        "messages": [
            {"role": "system", "content": (
                "你是医院 HRP 运营管理领域的专家，负责扩充高质量中文问答数据集。"
                "输出必须是合法 JSON 数组，元素为 {\"instruction\": 问题, \"output\": 专业回答}，"
                "回答 1-4 句话、术语准确、风格与示例一致，不要重复已有问题。"
            )},
            {"role": "user", "content": (
                f"主题：{topic}\n"
                f"请参考以下示例风格，生成 {n} 条新的问答对：\n"
                + json.dumps(examples, ensure_ascii=False)
            )},
        ],
        "response_format": {"type": "json_object"},
    }
    r = httpx.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload, timeout=120,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    # 兼容 {"data": [...]} 或直接 [...] 的返回
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = obj.get("data", obj.get("items", []))
        return [x for x in obj if isinstance(x, dict) and x.get("instruction") and x.get("output")]
    except Exception:
        m = re.search(r"\[.*\]", text, re.S)
        return json.loads(m.group(0)) if m else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/hrp_qa_seed.jsonl")
    ap.add_argument("--output", default="data/hrp_qa_expanded.jsonl")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--topic", default="医院HRP系统、物资SPD管理、成本核算、预算控制、绩效分配、资产设备、财务对账")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("请先设置环境变量 DEEPSEEK_API_KEY（示例：export DEEPSEEK_API_KEY=sk-xxx）")
        return

    seed = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    print(f"种子数据 {len(seed)} 条，目标扩到 {args.target} 条")

    expanded, seen = list(seed), {x["instruction"] for x in seed}
    batch = 15
    while len(expanded) < args.target:
        n = min(batch, args.target - len(expanded))
        try:
            new_items = call_llm(api_key, random.sample(seed, min(8, len(seed))), n, args.topic)
        except Exception as e:
            print(f"调用失败（{e}），30s 后重试...")
            time.sleep(30)
            continue
        fresh = [x for x in new_items if x["instruction"] not in seen]
        expanded.extend(fresh)
        seen.update(x["instruction"] for x in fresh)
        print(f"  已扩至 {len(expanded)}/{args.target}（本轮新增 {len(fresh)}）")
        time.sleep(2)

    with open(args.output, "w", encoding="utf-8") as f:
        for x in expanded[:args.target]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"完成：{args.output}，共 {args.target} 条（含种子）")


if __name__ == "__main__":
    main()
