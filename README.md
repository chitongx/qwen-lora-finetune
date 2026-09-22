# Qwen2.5 领域模型 LoRA/QLoRA 微调与部署

基于 **Qwen2.5-3B + QLoRA（Unsloth 加速）** 的领域大模型微调项目，
把通用模型"教会"医院 HRP 运营管理领域问答，产出可量化的训练前后对比指标，
并完整走通 **训练 → 评估 → GGUF 量化 → Ollama 部署 → OpenAI 兼容 API** 的工业链路。
全程 **免费 Colab T4 GPU** 即可跑完，适合作为 AI 应用开发工程师的第二个简历项目。

## 为什么做这个项目（面试叙事）

- 简历需要有 **LangChain/FAISS/Milvus/LoRA/微调** 这些 JD 高频词对应的**真实做过**的项目；
- 与项目一（RAG 知识库）形成互补：RAG 解决"外挂知识 + 实时更新"，微调解决"领域风格 + 私有部署"，
  面试常问"RAG 和微调怎么选"，两个项目都有完整实践可以直接答；
- 数据集自建（HRP 医院运营主题，贴合你金蝶 HRP 工作背景），微调后效果有数字可讲。

## 交付内容

```
qwen-lora-finetune/
├── notebooks/
│   └── qwen_lora_finetune_colab.ipynb   # 完整可跑：安装→训练→评估→GGUF→Ollama
├── data/
│   └── hrp_qa_seed.jsonl                # 33 条高质量种子问答（HRP 领域，人工编写）
├── scripts/
│   ├── gen_notebook.py                  # 重新生成 notebook（改了内容后重跑）
│   └── expand_dataset.py                # 用 DeepSeek API 把种子数据扩到 500+ 条
└── README.md
```

## 使用步骤（预计总耗时 1 小时，全免费）

### 1. 跑通训练（Colab 免费 T4）

1. 打开 [Google Colab](https://colab.research.google.com)，上传 `notebooks/qwen_lora_finetune_colab.ipynb`；
2. 菜单 `Runtime → Change runtime type → T4 GPU`；
3. 依次运行所有单元格：
   - 第 0 步：安装 Unsloth（约 1-2 分钟）
   - 第 1-2 步：加载 Qwen2.5-3B-Instruct 4bit + 配置 LoRA
   - 第 3 步：数据准备（内置 5 条示例可直接跑通；正式用 `data/hrp_qa_seed.jsonl` 上传替换）
   - 第 4 步：QLoRA 训练（60 步约 10-20 分钟）
   - 第 5 步：保存适配器
   - 第 6 步：**训练前后对比评估**，自动输出 keyword hit rate 与提升
   - 第 7 步：导出 GGUF 4bit 模型（约 2GB，下载保存）

### 2. 扩数据（可选，建议正式版做）

```bash
export DEEPSEEK_API_KEY=sk-xxx
python scripts/expand_dataset.py --target 500
# 产物：data/hrp_qa_expanded.jsonl（含 33 条种子 + 467 条扩充）
# 建议人工抽查 10%，把答错的删掉再训练
```

### 3. 本地部署（Ollama）

见 notebook 第 8 步：写 Modelfile → `ollama create hrp-qa -f Modelfile` → 调 OpenAI 兼容 API。
部署后可以把 `LLM_API_BASE=http://localhost:11434/v1` 配到项目一 RAG 系统里当生成端。

## 评估设计（为什么这个数字能写简历）

- 固定 8 个领域测试问题，每个标注期望关键词（如"三单匹配"→ 采购订单/入库单/发票）；
- base 模型与微调后模型用**完全相同**的 prompt、temperature 生成回答；
- 统计 keyword hit rate 并打印逐题对比，前后差距即"领域知识注入"的效果证明；
- 跑完把 notebook 第 6 步输出的真实数字填进简历模板（见 notebook 最后一节）。

## 面试追问准备

- **LoRA 原理**：冻结原模型，训练低秩矩阵 ΔW=A·B（r<<d），推理时 W'=W+ΔW 合并，零推理开销；
- **QLoRA 与 LoRA**：QLoRA = 4bit NF4 量化基础模型 + 双重量化 + 分页优化器，单卡 16GB 可微调 7B/13B；
- **为什么 LoRA 不全参微调**：显存成本低一个数量级、多任务可插拔、降低灾难性遗忘；
- **数据量**：领域 SFT 几百~几千条高质量即可，质量 > 数量；
- **RAG vs 微调**：知识实时更新/外挂文档 → RAG；固定领域风格、私有部署、隐式知识 → 微调。

## 进阶路线

1. 用 `expand_dataset.py` 扩到 500+ 条再训，指标会更好；
2. 换 Qwen2.5-7B（T4 QLoRA 仍可跑）；
3. 评估升级为 lm-evaluation-harness 或人工盲测打分；
4. 学习 DPO 对齐，进一步优化回答风格。
