# Qwen3 领域模型 LoRA/QLoRA 微调与部署

基于 **Qwen3-14B + QLoRA（Unsloth 加速）** 的领域大模型微调项目，
把通用模型"教会"医院 HRP 运营管理领域问答，产出可量化的训练前后对比指标，
并完整走通 **训练 → 评估 → GGUF 量化 → Ollama 部署 → OpenAI 兼容 API** 的工业链路。
全程 **免费 Colab T4 GPU** 即可跑完（Unsloth 官方确认 Qwen3-14B 在 T4 16GB 上舒适可跑），适合作为 AI 应用开发工程师的第二个简历项目。

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
├── docker-compose.yml                   # Ollama 部署（微调模型上线）
├── Modelfile                            # Ollama 模型模板（Qwen ChatML）
├── deploy.sh                            # 一键部署 + 验收
├── hf_space/                            # HuggingFace Spaces 公网 Chat 演示（Gradio）
└── README.md
```

## 上线部署（两条路径，任选其一）

### 路径 A：Docker + Ollama 部署微调模型（本机 / 云服务器）

```bash
# 1. 把 Colab 第 7 步导出的 model-Q4_K_M.gguf（约 2GB）放到 ./models/ 目录
# 2. 一键部署并验收：
./deploy.sh
# 3. 完成后得到 OpenAI 兼容接口：http://localhost:11434/v1 （model=hrp-qa）
#    可接入项目一 RAG 的 .env：LLM_API_BASE=http://localhost:11434/v1, LLM_MODEL=hrp-qa
```

云服务器部署后把 `localhost` 换成公网 IP 即可对外提供 API。

### 路径 B：Streamlit Community Cloud 免费公网 Chat（简历可直接放链接）

> 说明：HuggingFace Spaces 的 Docker/Gradio 托管 2026 年起需 PRO 订阅，**免费公网推荐 Streamlit Community Cloud**。

1. Colab 第 5 步把 LoRA 推送到 HF：`model.push_to_hub("你的用户名/hrp-qa-qwen-lora")`；
2. 打开 [streamlit.io/cloud](https://streamlit.io/cloud) → 用 GitHub 登录 → Create app → 选本仓库（`chitongx/qwen-lora-finetune`）→ Main file `streamlit_app.py` → Deploy；
3. 在 Settings → **Secrets** 添加：`BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct`（CPU 免费实例建议 0.5B）、`LORA_REPO=你的用户名/hrp-qa-qwen-lora`（留空则演示 base 模型）；
4. 约 3-5 分钟构建完成，得到公网地址：`https://qwen-lora-finetune.streamlit.app`，可直接放进简历。

> 提示：`hf_space/` 目录为 HuggingFace PRO 用户的 Gradio 备选方案（免费账号托管 Gradio Space 需付费）；免费部署优先走 Streamlit。

## 使用步骤（预计总耗时 1 小时，全免费）

### Colab 验收清单（跑完逐项打勾，即达到"可验收"程度）

- [ ] `Runtime → Change runtime type → T4 GPU`，第 0 步打印出 GPU 名称
- [ ] 第 0-2 步：Unsloth 安装成功，`model.print_trainable_parameters()` 显示可训练参数 <1%（约 0.4%）
- [ ] 第 3 步：数据集加载打印 `共 N 条训练数据`
- [ ] 第 4 步：训练正常启动并完成（T4 上 60 步约 10-20 分钟，日志出现 loss 递减）
- [ ] 第 5 步：`lora_model/` 目录生成 adapter 权重
- [ ] 第 6 步：打印出"微调后 keyword hit rate"与"微调前 keyword hit rate"及提升百分比 → **把这两个数字记下来**
- [ ] 第 7 步：`gguf_model/` 生成 `model-Q4_K_M.gguf`（约 2GB），下载到本地
- [ ] 第 8 步（可选）：本机装 Ollama，`ollama create hrp-qa -f Modelfile` 后能对话

跑完后把第 6 步的真实数字发我，我会同步更新简历上的微调项目描述。

### 1. 跑通训练（Colab 免费 T4）

1. 打开 [Google Colab](https://colab.research.google.com)，上传 `notebooks/qwen_lora_finetune_colab.ipynb`；
2. 菜单 `Runtime → Change runtime type → T4 GPU`；
3. 依次运行所有单元格：
   - 第 0 步：安装 Unsloth（约 1-2 分钟）
   - 第 1-2 步：加载 Qwen3-14B 4bit + 配置 LoRA（约 3-5 分钟下载模型）
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
2. 训练已升级到 Qwen3-14B（T4 上限）；更大需 Colab Pro（A100）跑 Qwen3-32B；
3. 评估升级为 lm-evaluation-harness 或人工盲测打分；
4. 学习 DPO 对齐，进一步优化回答风格。
