"""Generate the self-contained Colab notebook with only Python's standard library."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
seed = [json.loads(line) for line in (ROOT / "data/hrp_qa_seed.jsonl").read_text().splitlines() if line.strip()]
assert len(seed) == 33 and len({row["instruction"] for row in seed}) == 33
cells = []


def cell(kind, source):
    item = {"cell_type": kind, "id": f"hrp-{len(cells):02d}", "metadata": {},
            "source": dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """
# HRP 领域 LoRA 实验 · Qwen3.5-4B

目标是完成可复核的小样本实验：33 条原始种子 → 固定 25 条训练 / 8 条留出评估 → 同一模型先测 baseline → 16bit LoRA → 再测 tuned。
当前仓库没有 GPU 训练记录、adapter 或真实提升数字。数据较小，结果仅描述这 8 题，不能证明生产效果；允许持平或下降。

在 Colab 选择 **Runtime → Change runtime type → T4 GPU**，从上到下运行。默认数据已经完整内嵌，无需上传或输入密钥。
适配器、完整回答、命中分子/分母、数据 hash、环境和 GPU 峰值会先打包下载；GGUF 是最后的可选步骤。
免费 GPU 配额与运行时可用性由 Colab 决定，不保证时长。训练、推理与部署统一使用 Qwen3.5-4B。遵循官方建议，不做 Qwen3.5 的 4bit QLoRA。
""")
cell("markdown", """
## 0 · GPU 检查和依赖

安装方式与 [Unsloth 官方 Qwen3.5 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb) 对齐（核对日期 2026-10-10），本实验仅训练文本。
固定关键兼容版本：PyTorch 2.8、xformers 0.0.32.post2、Transformers 5.2.0、TRL 0.22.2。先在子进程检查 GPU，再安装，避免已导入的旧 PyTorch 干扰新版本。
环境版本写入结果 JSON。首次 GPU 完整训练仍待实测；遇到不支持的运行时先停止，换新的 Colab T4 会话。
""")
cell("code", """
import os
import subprocess
import sys

try:
    probe = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    raise RuntimeError("未检测到 CUDA GPU；请先在 Colab 切换到 T4 GPU，不会开始下载模型。")
if int(probe.stdout.strip().splitlines()[0].rsplit(",", 1)[1]) < 14 * 1024:
    raise RuntimeError("本实验需要至少约 14 GiB CUDA 显存，请使用 Colab T4。")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "uv"])
install = [sys.executable, "-m", "uv", "pip", "install", "--system"]
subprocess.check_call(install + ["torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0",
                               "triton>=3.3.0", "bitsandbytes", "xformers==0.0.32.post2",
                               "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
                               "unsloth[base] @ git+https://github.com/unslothai/unsloth"])
subprocess.check_call(install + ["--no-deps", "torchcodec==0.7.0", "tokenizers>=0.22.0,<=0.23.0", "trl==0.22.2"])
subprocess.check_call(install + ["transformers==5.2.0"])
subprocess.check_call(["unsloth", "install-kernels"])
subprocess.check_call(install + ["--no-deps", "--upgrade", "torchao>=0.16.0"])
print("GPU:", probe.stdout.strip())
""")
cell("markdown", """
## 1 · 固定训练与留出集

8 道原始问题不进入训练集。相邻主题仍可能重叠，此划分只防止问题逐字泄漏，不代表跨医院或跨领域泛化。
关键词来自留出答案，按区分大小写的字面包含计数；这是代理指标，不等同于事实准确率。
""")
cell("code", """
# DATA: embedded repository seed; no optional upload changes the split silently.
import hashlib
import json
from pathlib import Path

seed = """ + repr(seed) + """
heldout_keywords = {
    0: ["医院资源规划", "HRP", "业务财务一体化"],
    4: ["SPD", "院内物流", "配送"],
    8: ["VMI", "供应商管理库存", "补货"],
    10: ["公开招标", "竞争性磋商", "询价"],
    11: ["三单", "采购订单", "入库单", "发票"],
    19: ["RBRVS", "工作量", "积分"],
    24: ["预算", "控制", "特批"],
    27: ["年限平均法", "工作量法", "折旧"],
}
data = [row for i, row in enumerate(seed) if i not in heldout_keywords]
eval_set = [dict(seed[i], keywords=keywords, source=f"data/hrp_qa_seed.jsonl:{i + 1}")
            for i, keywords in heldout_keywords.items()]
assert len(data) == 25 and len(eval_set) == 8
assert not ({row["instruction"] for row in data} & {row["instruction"] for row in eval_set})

def data_hash(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

dataset_manifest = {"seed_count": len(seed), "train_count": len(data), "heldout_count": len(eval_set),
                    "seed_sha256": data_hash(seed), "train_sha256": data_hash(data),
                    "heldout_sha256": data_hash(eval_set), "heldout_seed_indices": list(heldout_keywords)}
artifacts = Path("artifacts")
artifacts.mkdir(exist_ok=True)
for name, rows in [("train", data), ("heldout", eval_set)]:
    (artifacts / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows), encoding="utf-8")
(artifacts / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(dataset_manifest, ensure_ascii=False, indent=2))
""")
cell("markdown", """
## 2 · 加载唯一基础模型

[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)：使用 tokenizer 原生 chat template，训练与评估都传 enable_thinking=False，只输出直接回答。
固定短上下文 1024，保留 T4 显存余量。实际显存和依赖版本会记录，不预先承诺参数占比或速度提升。
""")
cell("code", """
from unsloth import FastLanguageModel
import torch
from importlib.metadata import version
from datetime import datetime, timezone
import time

MODEL_ID = "Qwen/Qwen3.5-4B"
SEED = 3407
max_seq_length = 1024
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID, max_seq_length=max_seq_length,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16,
    load_in_4bit=False,
)
gpu = torch.cuda.get_device_properties(0)
environment = {name: version(name) for name in
               ["torch", "unsloth", "unsloth_zoo", "transformers", "trl", "peft", "datasets", "bitsandbytes", "accelerate", "xformers"]}
gpu_info = {"name": gpu.name, "total_memory_gib": gpu.total_memory / 1024**3,
            "cuda_version": torch.version.cuda}
memory = {}

def memory_snapshot():
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}

print("Base model:", MODEL_ID)
""")
cell("markdown", """
## 3 · 先运行 baseline

同样的 8 题、原生模板、greedy 解码（do_sample=False）、256 新 token 上限用于前后评估。
不使用随机采样，完整保留输出；到 token 上限的回答会记入生成记录。首次加载后尚未注入 LoRA，因此 baseline 是真正未训练的基础模型。
""")
cell("code", """
tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
generation_log = {}

def generate(model, tokenizer, question, max_new=256):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
        enable_thinking=False,
    ).to("cuda")
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    output_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generation_log[question] = {"new_tokens": len(output_ids), "at_token_limit": len(output_ids) >= max_new}
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def evaluate(model, tokenizer):
    hits, total, rows = 0, 0, []
    for item in eval_set:
        answer = generate(model, tokenizer, item["instruction"])
        matched = [keyword for keyword in item["keywords"] if keyword in answer]
        hits += len(matched)
        total += len(item["keywords"])
        rows.append(dict(item, answer=answer, matched_keywords=matched,
                         hit_keywords=len(matched), total_keywords=len(item["keywords"])))
    return {"hit_keywords": hits, "total_keywords": total, "rate": hits / total if total else 0, "rows": rows}

FastLanguageModel.for_inference(model)
torch.cuda.reset_peak_memory_stats()
base_started = time.perf_counter()
base_metrics = evaluate(model, tokenizer)
base_metrics["seconds"] = time.perf_counter() - base_started
base_metrics["generation"] = dict(generation_log)
memory["baseline"] = memory_snapshot()
(artifacts / "baseline.json").write_text(json.dumps(base_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"baseline: {base_metrics['hit_keywords']}/{base_metrics['total_keywords']} = {base_metrics['rate']:.2%}")
""")
cell("markdown", """
## 4 · r=16 LoRA，完整 3 个 epoch

训练 25 条，batch=1、梯度累积=8，学习率 2e-4；不再用 40 个 optimizer step 冒充 3 个 epoch。
只对 assistant 输出计算损失。回答模板由 tokenizer 生成，长度超限时停止，避免默默截断训练答案。
""")
cell("code", """
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

model = FastLanguageModel.get_peft_model(
    model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=SEED,
)
FastLanguageModel.for_training(model)
trainable, total_parameters = model.get_nb_trainable_parameters()
parameter_info = {"trainable": trainable, "total": total_parameters, "trainable_ratio": trainable / total_parameters}
texts = [tokenizer.apply_chat_template(
    [{"role": "user", "content": row["instruction"]}, {"role": "assistant", "content": row["output"]}],
    tokenize=False, add_generation_prompt=False, enable_thinking=False,
) for row in data]
lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
if max(lengths) > max_seq_length:
    raise RuntimeError("训练样本超过 max_seq_length，请检查模板/数据后重跑，不能静默截断。")
dataset = Dataset.from_dict({"text": texts})
use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text", max_length=max_seq_length, packing=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=3, learning_rate=2e-4, warmup_ratio=0.1,
        logging_steps=1, output_dir="outputs", save_strategy="no",
        optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
        seed=SEED, data_seed=SEED, bf16=use_bf16, fp16=not use_bf16,
        report_to="none",
    ),
)
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\\n",
                                response_part="<|im_start|>assistant\\n")
assert all(any(label != -100 for label in row["labels"]) for row in trainer.train_dataset), "assistant labels 全被屏蔽"
torch.cuda.reset_peak_memory_stats()
training_result = trainer.train()
memory["training"] = memory_snapshot()
adapter_dir = artifacts / "lora_adapter"
model.save_pretrained(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))
(artifacts / "training_log.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")
print("Adapter 已保存；真实训练统计:", training_result.metrics)
""")
cell("markdown", """
## 5 · 训练后评估，并先保存完整证据包

仅在这步完成后才有真实的前后数字。报告百分点变化，不把两模型能力差异归因于微调。
ZIP 包含 adapter、完整答案、训练日志、25/8 数据、数据 hash 和资源记录；不含任何认证信息。
""")
cell("code", """
import shutil

FastLanguageModel.for_inference(model)
generation_log.clear()
torch.cuda.reset_peak_memory_stats()
tuned_started = time.perf_counter()
tuned_metrics = evaluate(model, tokenizer)
tuned_metrics["seconds"] = time.perf_counter() - tuned_started
tuned_metrics["generation"] = dict(generation_log)
memory["tuned"] = memory_snapshot()
result = {
    "schema_version": 1, "model_id": MODEL_ID,
    "model_revision": getattr(model.config, "_commit_hash", None),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": dataset_manifest, "base": base_metrics, "tuned": tuned_metrics,
    "delta_percentage_points": (tuned_metrics["rate"] - base_metrics["rate"]) * 100,
    "training": dict(training_result.metrics, seed=SEED, epochs=3, batch_size=1,
                     gradient_accumulation_steps=8, lora_r=16, max_seq_length=max_seq_length,
                     method="LoRA", load_in_4bit=False),
    "parameters": parameter_info, "environment": environment, "gpu": gpu_info, "memory": memory,
    "generation_settings": {"do_sample": False, "max_new_tokens": 256, "enable_thinking": False},
    "limitations": "33 seeds / 25 training / 8 held-out: keyword inclusion is a small-sample proxy, not factual accuracy or production validation.",
}
(artifacts / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
archive = shutil.make_archive("hrp_qwen35_4b_artifacts", "zip", str(artifacts))
print(f"base={base_metrics['rate']:.2%}; tuned={tuned_metrics['rate']:.2%}; change={result['delta_percentage_points']:+.2f} 个百分点")
print("已保存证据包:", archive)
from google.colab import files
files.download(archive)
""")
cell("markdown", """
## 6 · 可选：GGUF 导出

默认跳过。先下载上一节 ZIP，再按需要改开关。GGUF 合并/量化可能额外消耗 RAM、磁盘、时间；失败不影响已保存 adapter 与结果。
不上传 Hugging Face、GitHub 或任何账户。需要发表模型时使用平台自己的安全登录界面，别把密钥写进 notebook。
""")
cell("code", """
RUN_GGUF_EXPORT = False
if RUN_GGUF_EXPORT:
    model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
    print("GGUF 已写入 gguf_model/；从 Colab 文件面板下载实际生成的 .gguf 文件。")
else:
    print("已跳过可选 GGUF；训练证据包已在上一步保存。")
""")
cell("markdown", """
## 7 · 结果演示与云端部署

免费 Streamlit Cloud 只展示 `evaluation.json` 的完整前后结果，不加载模型权重。将实测文件放入仓库 `results/evaluation.json` 后才会有真实结果。
新问题对话由云端 GPU 加载同一个 Qwen3.5-4B 基座和本次 adapter，网页只调用云端接口。本机不承担模型推理。
云端只使用实际可用的免费配额，部署前核验账号资格、额度及费用上限；记录页不会用其他模型冒充本次微调模型。

简历仅在真实训练、artifact 和部署验收完成后写“完成微调”及具体命中率。8 题样本需明确分母；GGUF / Ollama 未实测之前不写已部署。
""")

notebook = {
    "cells": cells,
    "metadata": {"colab": {"provenance": [], "name": "Qwen3_5_4B_HRP_LoRA"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
output = ROOT / "notebooks/qwen_lora_finetune_colab.ipynb"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"已生成: {output}")
