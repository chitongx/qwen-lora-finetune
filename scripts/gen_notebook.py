"""生成 qwen_lora_finetune_colab.ipynb（用 nbformat 构造，保证 JSON 合法可打开）。"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "colab": {"provenance": [], "name": "Qwen_HRP_LoRA_FineTune"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}
cells = []

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# 0 标题
cells.append(md("""# Qwen2.5 领域模型 LoRA/QLoRA 微调（HRP 医院运营问答助手）

> 目标：用 **免费 Colab T4 GPU** 微调 Qwen2.5-3B，让它"学会"医院 HRP 运营管理领域的专业问答，
> 并通过**训练前后对比评估**量化效果（简历可写的结果论指标）。
>
> 链路：数据准备 → QLoRA 训练（Unsloth 加速）→ 前后评估对比 → 导出 GGUF → Ollama 本地部署
>
> **重要**：菜单栏 `Runtime → Change runtime type → T4 GPU`，确认已连接 GPU 后再往下跑。"""))

# 1 安装
cells.append(md("## 第 0 步 · 安装依赖（约 1-2 分钟）"))
cells.append(code("""# Unsloth：训练提速 2 倍、显存减少 70%，是 Colab 免费微调的首选
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes xformers

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "未检测到 GPU，请先切换运行时为 T4")"""))

# 2 加载模型
cells.append(md("## 第 1 步 · 加载基础模型（4bit 量化，省显存）"))
cells.append(code("""from unsloth import FastLanguageModel
import torch

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",   # 可换 0.5B/1.5B/7B；7B 需 4bit+更长训练时间
    max_seq_length=max_seq_length,
    dtype=None,            # 自动检测（T4 用 fp16）
    load_in_4bit=True,     # 4bit 量化，显存从 ~7GB 降到 ~2GB
)
print("模型加载完成")"""))

# 3 LoRA 配置
cells.append(md("""## 第 2 步 · 配置 LoRA（只训练少量适配器参数）

LoRA 原理（面试必问）：冻结原模型全部参数，只在注意力与 FFN 层旁边加低秩矩阵 ΔW = A·B，
只训练 ΔW。`r` 是秩，控制适配器容量；`target_modules` 覆盖所有线性层，效果优于只调 q/v。"""))
cells.append(code("""model = FastLanguageModel.get_peft_model(
    model,
    r=16,                        # LoRA 秩（8~32 常见）
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",   # 显存换速度的关键
    random_state=3407,
)
model.print_trainable_parameters()  # 应远小于总参数（通常 <1%）"""))

# 4 数据准备
cells.append(md("""## 第 3 步 · 准备训练数据

两种方式任选：
- **A（推荐）**：把项目里 `data/hrp_qa_seed.jsonl` 上传到 Colab（左侧文件区拖入），或放在你的 GitHub 仓库后用 URL 下载；
- **B（快速试跑）**：直接用下面内置的 5 条示例数据验证全流程。

数据格式为 `{"instruction": 问题, "output": 标准回答}`（ShareGPT 风格），训练时统一转成 Qwen 的 ChatML 模板。"""))
cells.append(code("""import json

# ===== 方式 A：从上传文件加载 =====
# !pip install -q gdown  # 如需从 Google Drive 下载
# data = [json.loads(l) for l in open("hrp_qa_seed.jsonl", encoding="utf-8")]

# ===== 方式 B：内置示例（先用它跑通全流程）=====
seed = [
    {"instruction": "什么是HRP系统？", "output": "HRP（Hospital Resource Planning，医院资源规划）是医院运营管理的核心信息化平台，将人、财、物、信息四大资源统一管理，覆盖财务、物资、资产、人力、成本、预算、绩效等业务域，实现业务财务一体化和三流合一。"},
    {"instruction": "什么是SPD模式？", "output": "SPD（Supply, Processing, Distribution）是院内物流配送模式，由专业服务商在医院设中心库，按科室消耗定期补货，直接配送到科室消耗点，实现用后结算、零库存管理。"},
    {"instruction": "什么是三单匹配？", "output": "三单匹配指采购订单、入库单、发票三者一致性核对，只有三单匹配的应付账款才能进入付款排程，防止多付、错付与无依据付款。"},
    {"instruction": "医院采购有哪几种方式？", "output": "公开招标、邀请招标、竞争性磋商、询价、单一来源采购、网上竞价等，达到政府采购限额标准以上的项目须按政府采购程序执行。"},
    {"instruction": "什么是VMI？", "output": "VMI（Vendor Managed Inventory，供应商管理库存）是供应商根据医院库存消耗情况主动补货的模式，医院按实际消耗结算，降低采购与库存管理成本。"},
]
data = seed
print(f"共 {len(data)} 条训练数据")
print("示例：", json.dumps(data[0], ensure_ascii=False))"""))

# 5 格式化 + 训练
cells.append(md("""## 第 4 步 · 训练（QLoRA + SFT，T4 上约 10-30 分钟）

关键超参：`per_device_train_batch_size=2` + `gradient_accumulation_steps=4`（等效 batch 8）、
学习率 `2e-4`（LoRA 通常比全参微调高一个数量级）、`bf16`。"""))
cells.append(code("""from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# ShareGPT 格式 -> Qwen ChatML 模板
EOS_TOKEN = tokenizer.eos_token

def formatting_func(examples):
    texts = []
    for ins, out in zip(examples["instruction"], examples["output"]):
        text = f"<|im_start|>user\\n{ins}<|im_end|>\\n<|im_start|>assistant\\n{out}<|im_end|>"
        texts.append(text)
    return {"text": texts}

dataset = Dataset.from_list(data).map(formatting_func, batched=True)

# T4 不支持 bf16（Turing 架构），A100/L4 等新卡才支持；自动判断避免训练直接报错
use_bf16 = torch.cuda.is_bf16_supported()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=60,                 # 示例跑 60 步；正式训练建议 1-3 个 epoch
        logging_steps=5,
        output_dir="outputs",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        bf16=use_bf16,
        fp16=not use_bf16,
    ),
)
trainer.train()
print("训练完成 ✅")"""))

# 6 保存
cells.append(md("## 第 5 步 · 保存 LoRA 适配器（可上传 HuggingFace 留档）"))
cells.append(code("""# 保存适配器到本地
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("适配器已保存到 lora_model/")

# 可选：上传到 HuggingFace（面试时可展示训练产物）
# from huggingface_hub import notebook_login
# notebook_login()
# model.push_to_hub("你的用户名/hrp-qa-qwen-lora", tokenizer=tokenizer)"""))

# 7 评估
cells.append(md("""## 第 6 步 · 训练前后对比评估（简历核心数字）

对同一批测试问题，分别用 **微调前（base）** 和 **微调后（tuned）** 生成回答，
统计回答是否命中期望关键词（keyword hit rate），量化"领域知识注入"的效果。"""))
cells.append(code("""from unsloth import FastLanguageModel
import torch

# ---- 测试集：问题 + 期望出现的关键词 ----
eval_set = [
    {"q": "什么是HRP系统？", "kw": ["医院资源规划", "HRP", "业务财务一体化"]},
    {"q": "什么是SPD模式？", "kw": ["SPD", "院内物流", "配送"]},
    {"q": "什么是三单匹配？", "kw": ["三单", "采购订单", "入库单", "发票"]},
    {"q": "医院采购有哪几种方式？", "kw": ["公开招标", "竞争性磋商", "询价"]},
    {"q": "什么是VMI？", "kw": ["VMI", "供应商管理库存", "补货"]},
    {"q": "RBRVS是什么？", "kw": ["RBRVS", "工作量", "积分"]},
    {"q": "预算控制怎么做？", "kw": ["预算", "控制", "特批"]},
    {"q": "固定资产怎么折旧？", "kw": ["年限平均法", "工作量法", "折旧"]},
]

def generate(model, tokenizer, question, max_new=128):
    model = FastLanguageModel.for_inference(model)
    prompt = f"<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new, temperature=0.3, top_p=0.9)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def evaluate(model, tokenizer):
    hits, total = 0, 0
    rows = []
    for item in eval_set:
        ans = generate(model, tokenizer, item["q"])
        hit = sum(1 for k in item["kw"] if k in ans)
        hits += hit; total += len(item["kw"])
        rows.append((item["q"], hit, len(item["kw"]), ans[:80].replace("\\n", " ")))
    return hits / total if total else 0, rows

# ---- 微调后（当前内存中的 model）----
tuned_rate, tuned_rows = evaluate(model, tokenizer)
print(f"微调后 keyword hit rate = {tuned_rate:.2%}")

# ---- 释放显存，再加载 base 模型做对比（避免 T4 16GB 溢出）----
del model
import gc; gc.collect()
torch.cuda.empty_cache()

base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct", max_seq_length=max_seq_length,
    dtype=None, load_in_4bit=True,
)
base_rate, base_rows = evaluate(base_model, base_tokenizer)
print(f"微调前 keyword hit rate = {base_rate:.2%}")
print(f"提升 = {tuned_rate - base_rate:+.2%}")

print("\\n===== 对比明细 =====")
for (q1, h1, t1, a1), (q2, h2, t2, a2) in zip(base_rows, tuned_rows):
    print(f"\\nQ: {q1}")
    print(f"  base : [{h1}/{t1}] {a1}")
    print(f"  tuned: [{h2}/{t2}] {a2}")"""))

# 8 导出 GGUF
cells.append(md("""## 第 7 步 · 合并权重并导出 GGUF（Ollama 可直接加载）"""))
cells.append(code("""# 合并 LoRA 到基础模型，导出 GGUF 4bit（q4_k_m，文件约 2GB）
model.save_pretrained_gguf(
    "gguf_model",
    tokenizer,
    quantization_method="q4_k_m",   # 可选 q8_0 精度更高但文件更大
)
print("GGUF 已导出到 gguf_model/")
# 用 !ls -lh gguf_model/ 查看生成的文件，下载或上传到自己的服务器"""))

# 9 Ollama 部署
cells.append(md('''## 第 8 步 · 本地部署（Ollama + OpenAI 兼容 API）

把导出的 `.gguf` 文件（如 `gguf_model/model-Q4_K_M.gguf`）放到本机/服务器后：

```bash
# 1. 安装 Ollama（https://ollama.com）
# 2. 写 Modelfile（和 gguf 文件同目录）
#    FROM ./model-Q4_K_M.gguf
#    TEMPLATE """{{ if .System }}<|im_start|>system
#    {{ .System }}<|im_end|>
#    {{ end }}<|im_start|>user
#    {{ .Prompt }}<|im_end|>
#    <|im_start|>assistant
#    """
#    PARAMETER temperature 0.3

# 3. 创建并运行
#    ollama create hrp-qa -f Modelfile
#    ollama run hrp-qa

# 4. 调用（Ollama 自带 OpenAI 兼容接口）
curl http://localhost:11434/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"hrp-qa","messages":[{"role":"user","content":"什么是HRP系统？"}]}'
```

部署后可直接接进你之前的 RAG 项目做生成端，或单独做"本地私有化问答"演示。'''))

# 10 简历
cells.append(md("""## 简历写法（结果论模板）

> **Qwen2.5 领域模型 LoRA 微调与部署**（Python / QLoRA / PEFT / Unsloth / GGUF / Ollama）
> - 自建医院 HRP 运营管理领域问答数据集（含数据清洗与模板化），基于 Qwen2.5-3B 采用 QLoRA
>   （4bit 量化 + LoRA r=16）在单卡 16GB 上完成高效微调，可训练参数占比 <1%；
> - 建立训练前后同测试集对比评估，领域关键词命中率由 X% 提升至 Y%（提升 Z%）；
> - 合并 LoRA 权重并导出 GGUF 4bit 量化模型，经 Ollama 本地部署并提供 OpenAI 兼容 API。

## 面试追问准备

- **LoRA 原理**：冻结原模型，训练低秩分解矩阵 ΔW = A·B（A∈R^{d×r}，B∈R^{r×d}），
  推理时 W' = W + ΔW 合并，不增加推理开销；
- **QLoRA 与 LoRA 区别**：QLoRA 把基础模型 4bit 量化（NF4）+ 分页优化器 + 双重量化，
  单卡 16GB 能微调 7B/13B；
- **为什么选 LoRA 而不是全参微调**：显存与成本低一个数量级、可插拔多任务切换、防灾难性遗忘；
- **数据量多少合适**：领域 SFT 起步几百到几千条高质量数据，数据质量 > 数量；
- **什么场景选微调而不是 RAG**：知识静态 + 需要固定风格/格式输出 + 私有化部署 + RAG 解决不了的隐式知识。

## 进阶路线

1. 数据扩增：用 DeepSeek API 把 33 条种子数据扩到 500+ 条（见项目 `scripts/expand_dataset.py`）；
2. 换成 Qwen2.5-7B（T4 上 QLoRA 仍可跑，效果更好）；
3. 评估升级：用 lm-evaluation-harness 或人工盲测打分替代关键词命中；
4. 对齐进阶：尝试 DPO（偏好优化）提升回答风格一致性。"""))

nb.cells = cells
out = Path(__file__).resolve().parent.parent / "notebooks" / "qwen_lora_finetune_colab.ipynb"
nbf.write(nb, str(out))
print(f"已生成: {out}")
