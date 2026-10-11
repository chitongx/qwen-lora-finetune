# HRP 领域 LoRA 实验：Qwen3.5-4B

使用医院 HRP 运营问答建立可复核的训练、评估与云端部署流程。当前基础模型统一为 **`Qwen/Qwen3.5-4B`**，训练方式为 16bit LoRA + Unsloth，默认面向免费 Colab T4。本机不承担模型推理。

**2026-10-11 已在免费 Colab T4 完成真实 LoRA 训练：关键词命中率 18/25（72%）→17/25（68%），下降 4 个百分点。** 完整逐题记录见 [`results/evaluation.json`](results/evaluation.json)。Adapter 已在 Colab 保存，下载和 HF 发布尚待确认；云端推理尚未部署，不声称微调改善了回答质量。

本轮使用 25 条训练数据、3 个 epoch、12 个优化步，训练耗时 249.59 秒，可训练参数 21,233,664 / 4,560,499,200（0.47%），训练峰值已分配显存 8.84 GiB。Unsloth 自动关闭 FP16/BF16 混合精度标志；评估后记录的模型参数张量同时含 FP16 与 FP32，不能描述为所有权重纯 FP32。基础模型 revision、实际标志、数据指纹与 adapter SHA256 均保存在原始记录中。

逐题变化：SPD 少命中“配送”，三单匹配使用“收货单”而未命中“入库单”，采购题新增“询价”命中；净减少 1 个关键词。基座 8/8、微调后 2/8 回答达到 256 token 上限；微调后采购回答出现重复。字面指标无法独立判断事实质量，此 adapter 尚不满足作为更优问答模型的上线验收。后续应补充独立的新问题与人工事实/重复检查，避免反复围绕这 8 题调参后仍把它们称为未见过的最终评估集。

## 为什么替换 14B

本项目的约束是免费 T4、免费云端演示和小样本数据。选择 Qwen3.5-4B 是针对可用资源的工程取舍，不代表它在本领域已被证明优于 14B，或是市场最强模型。

[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)提供模型说明。[Unsloth 官方微调指南](https://unsloth.ai/docs/models/qwen3.5/fine-tune)列出 4B 的免费 Colab 路线与约 10GB 的 16bit LoRA 显存需求，明确不推荐 Qwen3.5 的 4bit QLoRA。本实验仅处理文本，训练与推理原生模板统一关闭 thinking。资料核对日期：2026-10-10；本次 T4 已完成训练，不能据此承诺其他免费 GPU 会话必然可用。

## 数据、评估和证据

参考项目核对日期 2026-10-10：[Unsloth](https://github.com/unslothai/unsloth) 77,682 stars，[LlamaFactory](https://github.com/hiyouga/LlamaFactory) 75,397 stars。采用 Unsloth 官方 Qwen3.5 安装与模型模板；借鉴 LlamaFactory 的训练/评估数据分离、assistant-only loss、训练日志与 adapter 导出思路。star 数是当日快照。本项目保留短 notebook，不增加 WebUI 或第二套训练框架，先取得当前 25/8 实验的真实结果。

- 原始文件 `data/hrp_qa_seed.jsonl` 共 33 条，完整内嵌到 notebook，默认无需上传。
- 固定原始索引 `[0, 4, 8, 10, 11, 19, 24, 27]` 的 8 题作为留出集，其余 25 条用于训练。问题不重复，相关主题可能重叠，不据此宣称跨域泛化。
- 在同一个基础模型上先实际测 baseline，再加入 r=16 LoRA，训练完整 3 个 epoch，最后测 tuned。上下文 1024，batch 1，梯度累积 8，学习率 `2e-4`。
- 前后使用相同原生模板、`do_sample=False`、256 新 token 上限。完整回答和达到长度上限的记录都会保存。
- 指标为区分大小写的字面关键词包含率，总分母为 **25 个关键词 / 8 题**。它是小样本代理指标，不是回答准确率；允许训练后持平或下降。
- `evaluation.json` 保存逐题回答、命中分子/分母、百分点变化、数据 SHA256、训练参数、依赖版本、GPU 信息、训练日志和各阶段峰值显存。

## 在 Colab 完成真实实验

1. 将 `notebooks/qwen_lora_finetune_colab.ipynb` 上传到 [Colab](https://colab.research.google.com/)，选择 **Runtime → Change runtime type → T4 GPU**，从上到下运行。
2. GPU 检查先执行；没有 CUDA 或显存不足时停止，不下载模型。安装与官方 Qwen3.5 notebook 对齐，固定 `torch==2.8.0`、`xformers==0.0.32.post2`、`transformers==5.2.0`、`trl==0.22.2`。T4 的精度请求会由 Unsloth 调整，结果文件记录训练器的实际标志。
3. 完成第 5 节后下载自动生成的 **`hrp_qwen35_4b_artifacts.zip`**，包含 adapter、完整 `evaluation.json`、baseline、训练日志、训练/留出数据和指纹。
4. GGUF 导出在第 6 节，默认 `RUN_GGUF_EXPORT = False`。证据包下载后，可改为 `True` 再单独运行该节；导出可能额外需要 RAM、磁盘和时间。下载实际生成的 `.gguf`，不要假设固定文件名。

免费 GPU 的分配与配额由 Colab 决定，不承诺 40–60 分钟完成。不自动上传 HF 或 GitHub，不在代码、结果包或仓库写入任何认证信息。若未来需要发表 adapter，应通过平台自己的安全登录流程完成。

## 免费公网结果页

已部署的免费实验记录页面：[qwen35-hrp-lora.streamlit.app](https://qwen35-hrp-lora.streamlit.app/)。2026-10-11 已在浏览器确认显示 72%→68%、下降提示、前后完整回答与原始记录；此验收不包含新问题 GPU 推理。

`streamlit_app.py` 只读取结果 JSON，依赖仅为 Streamlit。免费云不加载 4B 模型，也不使用一个 4B adapter 去挂载 0.5B 基座。

本轮实测 `evaluation.json` 已发布到仓库 `results/evaluation.json`，由 [Streamlit Community Cloud](https://streamlit.io/cloud) 的 `main` 分支自动更新，入口为 `streamlit_app.py`。实验记录页无需设置 BASE_MODEL、LORA_REPO 或密钥。

结果页显示前后命中率、完整回答和原始记录；文件不存在时明确提示训练结果尚未载入。也可以上传 JSON 临时查看。页面校验模型、25/8 划分、同题同关键词和回答对应的命中计数，但上传文件本身不是训练真实性的独立证明。

**云端对话的客户端、API 与部署入口已加入；真实 GPU 加载和公网验收尚未完成。** 云端 GPU 必须加载同一个 Qwen3.5-4B 基座和本次 adapter。HF 创建页面目前要求 Gradio/Docker 付费；候选替代为 Modal Starter 的每月免费计算额度，但只有核验账号并将现金支出上限设置为 0 后才部署。额度耗尽时停止服务，不启用付费 Shared Endpoints。见[价格](https://modal.com/pricing)与[支出限制](https://modal.com/docs/guide/budgets)。本项目不重试已失败的隧道或公网端口路径。

### 云端接口接入（尚未部署）

`cloud/server.py` 在加载前检查 `evaluation.json`、adapter 权重及基座标识；缺少训练产物或基座不一致即停止。接口 `/v1/chat/completions` 只提供 `hrp-qwen35-4b`，支持真实 token 流和完整多轮消息。演示输入加输出上限 4096 token，单次输出最多 256 token。此检查不能替代 GPU 推理验收。

`cloud/modal_app.py` 使用 L4 的原生 BF16，空闲时缩容至零，最多一个容器，闲置保留 2 秒；仍会消耗启动、加载和推理的计算额度，不能仅靠这些参数保证免费。部署前必须在账户账单页确认适用免费余额、用量限额和 **现金支出上限 $0**。没有账户访问或免费额度时停止，不启动资源。

取得并解压真实证据包后，在已授权的 Modal 环境运行：

```bash
HRP_ARTIFACT_DIR=/absolute/path/to/unpacked/artifacts modal deploy -m cloud.modal_app
```

API 默认启用 [Modal Proxy Tokens](https://modal.com/docs/guide/webhook-proxy-auth)，未授权请求由平台拒绝。通过平台页面配置凭据，不写入仓库、notebook 或结果文件。Streamlit Cloud 的 Settings → Secrets 使用 `HRP_CLOUD_URL`（不带 `/v1` 的接口根地址）和 `HRP_CLOUD_API_KEY`（平台要求的 Token ID 与 Secret 以点连接）；不要把真实值放在公开示例中。

网页仅调用云端 API，首次请求可能较慢；额度不足或流中断时显示错误，不把半截回答保存为完整对话。同一接口可接入 RAG：`LLM_API_BASE` 为根地址加 `/v1`，`LLM_MODEL=hrp-qwen35-4b`，鉴权信息仅在平台界面配置。需要分别验收新问题、追问、RAG 流式回答与来源引用后才算上线。

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

当前可写“在免费 T4 基于 Qwen3.5-4B 完成领域 LoRA 微调，可训练参数占比 0.47%、训练峰值显存 8.84 GiB；建立同基座前后评估并公开完整逐题记录”。若填写效果，应明确 72%→68%、下降 4 个百分点，不能写提升。云端 API 和 RAG 集成各自验收之后才写对应完成结果。不能将 33 条数据写成 500 条；`scripts/expand_dataset.py` 为旧的可选扩增脚本，未运行，不纳入当前默认实验。
