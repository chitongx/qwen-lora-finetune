# HRP 领域 LoRA 实验：Qwen3.5-4B

使用医院 HRP 运营问答建立可复核的训练、评估与云端部署流程。当前基础模型统一为 **`Qwen/Qwen3.5-4B`**，训练方式为 16bit LoRA + Unsloth，默认面向免费 Colab T4。本机不承担模型推理。

**当前状态：工程准备和离线验证已完成，真实 GPU 训练尚未完成。仓库没有实测前后命中率、训练后的 adapter 或 GGUF，不能声称已经完成微调或部署。** 不以占位数据、其他基础模型的回答或本地检查替代训练证据。

## 为什么替换 14B

本项目的约束是免费 T4、免费云端演示和小样本数据。选择 Qwen3.5-4B 是针对可用资源的工程取舍，不代表它在本领域已被证明优于 14B，或是市场最强模型。

[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)提供模型说明。[Unsloth 官方微调指南](https://unsloth.ai/docs/models/qwen3.5/fine-tune)列出 4B 的免费 Colab 路线与约 10GB 的 16bit LoRA 显存需求，明确不推荐 Qwen3.5 的 4bit QLoRA。本实验仅处理文本，训练与推理原生模板统一关闭 thinking。资料核对日期：2026-10-10；T4 的实际兼容性与资源使用仍以本次运行结果为准。

## 数据、评估和证据

- 原始文件 `data/hrp_qa_seed.jsonl` 共 33 条，完整内嵌到 notebook，默认无需上传。
- 固定原始索引 `[0, 4, 8, 10, 11, 19, 24, 27]` 的 8 题作为留出集，其余 25 条用于训练。问题不重复，相关主题可能重叠，不据此宣称跨域泛化。
- 在同一个基础模型上先实际测 baseline，再加入 r=16 LoRA，训练完整 3 个 epoch，最后测 tuned。上下文 1024，batch 1，梯度累积 8，学习率 `2e-4`。
- 前后使用相同原生模板、`do_sample=False`、256 新 token 上限。完整回答和达到长度上限的记录都会保存。
- 指标为区分大小写的字面关键词包含率，总分母为 **25 个关键词 / 8 题**。它是小样本代理指标，不是回答准确率；允许训练后持平或下降。
- `evaluation.json` 保存逐题回答、命中分子/分母、百分点变化、数据 SHA256、训练参数、依赖版本、GPU 信息、训练日志和各阶段峰值显存。

## 在 Colab 完成真实实验

1. 将 `notebooks/qwen_lora_finetune_colab.ipynb` 上传到 [Colab](https://colab.research.google.com/)，选择 **Runtime → Change runtime type → T4 GPU**，从上到下运行。
2. GPU 检查先执行；没有 CUDA 或显存不足时停止，不下载模型。安装与官方 Qwen3.5 notebook 对齐，固定 `torch==2.8.0`、`xformers==0.0.32.post2`、`transformers==5.2.0`、`trl==0.22.2`。环境首次 GPU 集成仍需实测。
3. 完成第 5 节后下载自动生成的 **`hrp_qwen35_4b_artifacts.zip`**，包含 adapter、完整 `evaluation.json`、baseline、训练日志、训练/留出数据和指纹。
4. GGUF 导出在第 6 节，默认 `RUN_GGUF_EXPORT = False`。证据包下载后，可改为 `True` 再单独运行该节；导出可能额外需要 RAM、磁盘和时间。下载实际生成的 `.gguf`，不要假设固定文件名。

免费 GPU 的分配与配额由 Colab 决定，不承诺 40–60 分钟完成。不自动上传 HF 或 GitHub，不在代码、结果包或仓库写入任何认证信息。若未来需要发表 adapter，应通过平台自己的安全登录流程完成。

## 免费公网结果页

`streamlit_app.py` 只读取结果 JSON，依赖仅为 Streamlit。免费云不加载 4B 模型，也不使用一个 4B adapter 去挂载 0.5B 基座。

训练后把证据包里的实测 `evaluation.json` 放到仓库 `results/evaluation.json`，发布到 GitHub，再在 [Streamlit Community Cloud](https://streamlit.io/cloud) 创建 app，入口选择 `streamlit_app.py`。无需设置 BASE_MODEL、LORA_REPO 或密钥。

结果页显示前后命中率、完整回答和原始记录；文件不存在时明确提示训练结果尚未载入。也可以上传 JSON 临时查看。页面校验模型、25/8 划分、同题同关键词和回答对应的命中计数，但上传文件本身不是训练真实性的独立证明。

**目前这是实验结果展示页面，云端新问题对话仍在接入。** 云端 GPU 必须加载同一个 Qwen3.5-4B 基座和本次 adapter。HF 创建页面目前要求 Gradio/Docker 付费；候选替代为 Modal Starter 的每月免费计算额度，但只有核验账号并将现金支出上限设置为 0 后才部署。额度耗尽时停止服务，不启用付费 Shared Endpoints。见[价格](https://modal.com/pricing)与[支出限制](https://modal.com/docs/guide/budgets)。本项目不重试已失败的隧道或公网端口路径。

## 保留的本机 GGUF + Ollama 示例（非当前部署路线）

以下是原 Qwen3 的部署示例，尚未验证 Qwen3.5 的模板和 GGUF 兼容性；本次不把它作为完成项或云端部署替代。

macOS 建议安装 [原生 Ollama](https://ollama.com/download) 并启动，利用 Metal；Docker 在 Mac 上作为 CPU 备选。尚未生成训练后的 GGUF 时，部署脚本会停止，不下载其他模型冒充训练产物。

```bash
# 将唯一一个训练后导出的 .gguf 放入 models/，名称不限。
mkdir -p models
./deploy.sh

# Docker CPU 备选（需要 Docker 已安装）。
./deploy.sh docker

# 安装页面依赖后，在本机开启真实对话。
python3 -m pip install -r requirements.txt
HRP_LOCAL_OLLAMA=1 streamlit run streamlit_app.py
```

脚本生成与实际 GGUF 路径一致的 `Modelfile.generated`，导入为 `hrp-qa`，实际调用 `http://127.0.0.1:11434/api/chat` 验收非空完整回答，保存 `results/ollama_smoke.json`。ChatML 模板与页面传递完整多轮消息。Docker 端口仅绑定回环地址，不开放公网。

导入并验收之后才可在 RAG 项目配置 `LLM_API_BASE=http://127.0.0.1:11434/v1` 和 `LLM_MODEL=hrp-qa`。两项目位于不同容器时，需要另行明确容器网络；这里没有验证这种集成。

## 离线工程验证

```bash
python3 scripts/gen_notebook.py
python3 -m unittest discover -s tests -v
bash -n deploy.sh
```

生成器只用标准库，重新生成后与已保存 notebook 完全一致。离线检查执行实际数据划分、关键词统计和结果验证函数，同时检查 baseline 顺序、GPU fail-fast、模板与解码、证据先保存、云页面轻量化及回环部署。

这些检查不加载模型，也不证明 GPU 依赖、训练、GGUF 导出、Docker 推理或 HF 发布已经成功。

## 简历可写边界

当前可写“构建 LoRA 训练与前后评估流程”。取得真实证据后，才写“基于 Qwen3.5-4B 完成领域 LoRA 微调”，并填入实测分子/分母和百分点变化。云端 API 和 RAG 集成各自验收之后才写对应完成结果。不能将 33 条数据写成 500 条；`scripts/expand_dataset.py` 为旧的可选扩增脚本，未运行，不纳入当前默认实验。
