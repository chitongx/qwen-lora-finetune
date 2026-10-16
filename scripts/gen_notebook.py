"""Generate the self-contained Colab notebook with only Python's standard library."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
seed = [json.loads(line) for line in (ROOT / "data/hrp_qa_seed.jsonl").read_text().splitlines() if line.strip()]
assert len(seed) == 33 and len({row["instruction"] for row in seed}) == 33
backup_source = (ROOT / "artifact_backup.py").read_text().split('\nif __name__ == "__main__":', 1)[0]
evaluation_source = (ROOT / "evaluation_v2.py").read_text().split('\nif __name__ == "__main__":', 1)[0]
eval_v2_cases = [json.loads(line) for line in (ROOT / "data/eval_v2.jsonl").read_text().splitlines() if line.strip()]
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
仓库已保存首轮真实 GPU 记录：关键词命中率 72%→68%，没有证明微调提升。本 notebook 用于复现实验；新会话生成的是新产物，不能冒充原 adapter。数据较小，结果仅描述这 8 题，不能证明生产效果。

在 Colab 选择 **Runtime → Change runtime type → T4 GPU**，从上到下运行。默认数据已经完整内嵌，无需上传或输入密钥。
训练前必须由本人授权挂载 Google Drive，创建独立 run 目录并通过写入回读检查；失败即停止，不盲目重训。适配器和记录直接写入该目录，ZIP 在 Colab 本地校验后另存 Drive，再下载到本机作为第二副本；GGUF 是最后的可选步骤。
免费 GPU 配额与运行时可用性由 Colab 决定，不保证时长。训练、推理与部署统一使用 Qwen3.5-4B。遵循官方建议，不做 Qwen3.5 的 4bit QLoRA。
""")
cell("markdown", """
## 0 · GPU、持久保存预检与依赖

安装方式与 [Unsloth 官方 Qwen3.5 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb) 对齐（核对日期 2026-10-10），本实验仅训练文本。
固定关键兼容版本：PyTorch 2.8、xformers 0.0.32.post2、Transformers 5.2.0、TRL 0.22.2。先在子进程检查 GPU，再安装，避免已导入的旧 PyTorch 干扰新版本。
环境版本写入结果 JSON。首轮 T4 已完成训练；遇到不支持的运行时先停止，换新的 Colab T4 会话。
Google Drive 挂载会由平台向你请求权限，请本人判断并完成；不在 notebook 中填写或保存任何 Token、Cookie 或密钥。每次运行创建新目录，不覆盖首轮结果。看到“持久写入已验证”后才会安装依赖。
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
""" + backup_source + """

from google.colab import drive
drive.mount("/content/drive")
drive_home = Path("/content/drive/MyDrive")
if not drive_home.is_dir():
    raise RuntimeError("Google Drive 未完成挂载；停止，不开始安装或训练。")
backup_storage = drive_home / "HRP_LoRA_Backups"
backup_storage.mkdir(exist_ok=True)
artifacts = prepare_run(backup_storage)
backup_cache = Path("/content/hrp_backup_cache")
print("持久写入已验证，独立运行目录:", artifacts.parent)
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
assert_backup_ready(artifacts)
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
assert_backup_ready(artifacts)
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
assert_backup_ready(artifacts)
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
from collections import Counter
training_precision = {
    "fp16": trainer.args.fp16, "bf16": trainer.args.bf16,
    "parameter_tensor_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
}
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
ZIP 包含 adapter、完整答案、训练日志、25/8 数据、数据 hash、运行身份与逐文件 SHA256 manifest；仅打包明确允许的产物文件，不包含认证信息。Drive 与本地 ZIP 都通过一致性检查后才发起浏览器下载；浏览器下载完成仍需你在 Mac 上确认并校验。
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
                     method="LoRA", load_in_4bit=False, precision=training_precision),
    "parameters": parameter_info, "environment": environment, "gpu": gpu_info, "memory": memory,
    "audit": {"run_id": artifacts.parent.name,
              "adapter_sha256": hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()},
    "generation_settings": {"do_sample": False, "max_new_tokens": 256, "enable_thinking": False},
    "limitations": "33 seeds / 25 training / 8 held-out: keyword inclusion is a small-sample proxy, not factual accuracy or production validation.",
}
(artifacts / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
archive = build_verified_bundle(artifacts, backup_cache)
print(f"base={base_metrics['rate']:.2%}; tuned={tuned_metrics['rate']:.2%}; change={result['delta_percentage_points']:+.2f} 个百分点")
print("Drive 已校验备份:", artifacts.parent / archive.name)
print("Colab 本地已校验副本:", archive)
print("ZIP 与权重指纹:", verify_bundle(archive))
from google.colab import files
files.download(str(archive))
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
cell("markdown", """
## 8 · 面试现场提问（单独运行，不重训）

仅在本会话仍保留已训练的 `model`、tokenizer 与 adapter 时使用。将下一单元的 `RUN_INTERVIEW_QA` 改为 True，**只运行该单元**，输入面试官的新问题。
先临时禁用 adapter 生成基础模型回答，再恢复 adapter 生成微调回答；原生模板、关闭 thinking、greedy 和 256 token 上限与原实验一致。
不改写 `evaluation.json`，不更新固定 8 题命中率；现场回答需要人工检查事实、遗漏、重复与截断。这里的“基座”表示同一模型临时禁用 adapter。

这只是免费 Colab 临时 GPU 演示，不是持续在线 API。会话丢失时明确停止，不自动下载替代模型或重新训练。恢复真实产物后先核验 model revision / adapter SHA256；若重新训练，另存新的运行记录和产物。
""")
cell("code", """
def interview_compare(question):
    required = ("model", "tokenizer", "generate", "generation_log", "FastLanguageModel", "max_seq_length")
    if any(name not in globals() for name in required) or not getattr(model, "peft_config", None):
        raise RuntimeError("当前会话没有已训练的 adapter；请先恢复真实训练产物，不能仅运行本单元冒充模型就绪。")
    question = question.strip()
    if not question:
        raise ValueError("请输入一个 HRP 问题。")
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    if len(token_ids) + 256 > max_seq_length:
        raise ValueError("问题超过本实验上下文预算，请缩短问题；不会静默截断。")
    FastLanguageModel.for_inference(model)
    with model.disable_adapter():
        base_answer = generate(model, tokenizer, question)
        base_generation = dict(generation_log[question])
    tuned_answer = generate(model, tokenizer, question)
    return {"question": question, "base_answer": base_answer, "tuned_answer": tuned_answer,
            "base_generation": base_generation, "tuned_generation": dict(generation_log[question]),
            "kind": "live_interview", "note": "现场新问题；不计入原 8 题命中率。"}

RUN_INTERVIEW_QA = False
if RUN_INTERVIEW_QA:
    live_answer = interview_compare(input("输入面试现场 HRP 新问题："))
    print("基础模型（临时禁用 adapter）:\\n", live_answer["base_answer"])
    print("微调模型（已恢复 adapter）:\\n", live_answer["tuned_answer"])
    print("生成状态:", live_answer["base_generation"], live_answer["tuned_generation"])
    print(live_answer["note"])
else:
    print("现场提问默认关闭。保留模型的会话内改为 True，只运行本单元即可。")
""")
cell("markdown", """
## 9 · 独立扩展评估（默认关闭，不重训）

新增 28 道原领域知识的新表达、组合与反例题，**agent 编写、未经人工业务审阅、不加入训练**；它们不是外部知识或统计独立的专业评测集。
只有同一会话保留真实模型、对应 adapter 和已验证备份时才可改 `RUN_EVAL_V2=True`，只运行本单元。
先临时禁用 adapter 测基座，再恢复 adapter 测微调。关闭 thinking，greedy，256 新 token。56 次生成会额外消耗免费 GPU 额度；没有可用会话就保持关闭，不能填入臆造指标。

新报告单独保存为 `evaluation_v2.json`，基座阶段先保存为 `evaluation_v2_base.json`，不改首轮 `evaluation.json`、固定 8 题或训练数据。
报告保留完整回答、同义词组字面覆盖、字符四元组重复信号、长度上限信号和人工待审标记。字面覆盖不判断否定、流程顺序或事实正确性；长度触顶也不必然代表答案不完整。
完成后生成新的双副本 ZIP，旧 ZIP 保留。真实模型输出和人工业务复核都未执行前，不写新的模型效果数字。
""")
cell("code", evaluation_source + """

eval_v2_cases = """ + repr(eval_v2_cases) + """

def run_evaluation_v2():
    global archive
    required = ("model", "tokenizer", "generate", "generation_log", "FastLanguageModel", "artifacts", "archive", "backup_cache", "data_hash")
    if any(name not in globals() for name in required) or not getattr(model, "peft_config", None):
        raise RuntimeError("当前会话缺少真实 adapter 或已验证备份；保持关闭，不开始新训练。")
    assert_backup_ready(artifacts)
    if (artifacts / "evaluation_v2.json").exists():
        raise RuntimeError("此运行已有扩展评估记录；请保留原记录，不重复消耗 GPU 或覆盖。")
    evidence = json.loads((artifacts / "evaluation.json").read_text())
    adapter_hash = evidence["audit"]["adapter_sha256"]
    backup = verify_bundle(archive, expected_adapter_sha256=adapter_hash)
    if (sha256_file(artifacts / "lora_adapter" / "adapter_model.safetensors") != adapter_hash
            or getattr(model.config, "_commit_hash", None) != evidence["model_revision"]):
        raise RuntimeError("会话/备份的基座版本或 adapter 指纹不一致，停止扩展评估。")
    FastLanguageModel.for_inference(model)
    def observed_answer(question):
        answer = generate(model, tokenizer, question)
        return dict(generation_log[question], answer=answer, max_new_tokens=256)
    with model.disable_adapter():
        base_v2 = evaluate_cases(eval_v2_cases, observed_answer)
    (artifacts / "evaluation_v2_base.json").write_text(json.dumps(base_v2, ensure_ascii=False, indent=2), encoding="utf-8")
    tuned_v2 = evaluate_cases(eval_v2_cases, observed_answer)
    expanded = {"schema_version": 2, "source_run_id": backup["run_id"], "model_id": MODEL_ID,
        "model_revision": evidence["model_revision"], "adapter_sha256": adapter_hash,
        "eval_cases_sha256": data_hash(eval_v2_cases), "base_method": "same loaded model with adapter disabled",
        "generation_settings": {"do_sample": False, "max_new_tokens": 256, "enable_thinking": False},
        "prepared_data_unreviewed": True, "base": base_v2, "tuned": tuned_v2}
    assert_backup_ready(artifacts)
    (artifacts / "evaluation_v2.json").write_text(json.dumps(expanded, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = build_verified_bundle(artifacts, backup_cache)
    print("扩展评估已独立保存，所有题仍待人工业务复核:", artifacts / "evaluation_v2.json")
    print("新 Drive ZIP:", artifacts.parent / archive.name)
    return expanded

RUN_EVAL_V2 = False
if RUN_EVAL_V2:
    expanded_result = run_evaluation_v2()
    from google.colab import files
    files.download(str(archive))
else:
    print("扩展评估准备完成，尚未生成新模型结果；保持关闭直到真实模型与备份就绪。")
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
