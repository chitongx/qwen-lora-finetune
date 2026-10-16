# LoRA 持久备份与扩展评估

## 当前状态

2026-10-16 按文件名只读检查 Mac 的 Downloads 顶层，未发现同时含 HRP/Qwen/adapter 名称且以 ZIP 或 safetensors 结尾的候选备份；没有扫描其他文件内容或认证目录。原始 `results/evaluation.json` 的 72%→68% 保留不动，不能用它证明原 adapter 目前可下载。

本次已准备持久保存流程、离线校验工具及 28 道扩展题。**尚未在真实 Drive/GPU 上执行新流程，尚无扩展题的模型结果。** 挂载授权由本人在 Google 页面完成；代码不填写、收集或保存密码、Token、Cookie 或 API Key。

## 开始训练前

先核查是否已有备份：本机、Drive 或其他已授权模型存储中存在对应 adapter 时，优先校验和恢复，避免重复训练。没有可靠的持久输出位置时停止。

Notebook 第 0 节检查 T4 后，请本人完成 `drive.mount` 的 Google Drive 权限选择。代码在 `MyDrive/HRP_LoRA_Backups/` 下创建带 UTC 时间和随机后缀的独立 run，写入并回读检查文件；失败会在依赖安装、模型下载和训练前停止。模型加载与训练单元还会重新检查可写性。

建议先单独运行第 0 节，到 Drive 网页确认新 run 可见，再继续后续单元。每次是独立目录，不覆盖其他 run；原 25 条训练/8 条留出划分和首轮结果不变。**文件系统回读证明挂载目录可写，不能替代结束会话前在 Drive 网页独立查看和下载文件。**

## 训练后的两份副本

1. Adapter、tokenizer、数据划分、baseline、训练日志和对应 `evaluation.json` 直接写入此 run 的 Drive 目录。记录包含基座 revision 和实际权重 SHA256。
2. 第 5 节生成逐文件大小/SHA256 的 `manifest.json`，仅收录明确允许的训练与评估产物。先在 Colab 的 `/content/hrp_backup_cache/` 生成 ZIP，逐文件校验，再复制回 Drive 并再次核验两份一致。
3. 浏览器从 Colab 本地 ZIP 下载，不直接传输挂载目录中的 ZIP。下载提示出现**不等于 Mac 已取得第二副本**：须在 Downloads 确认文件存在，执行下面的只读校验，并对照 notebook 输出的 ZIP SHA256、adapter SHA256。
4. 在 Drive 网页确认 run、adapter、`evaluation.json`、`manifest.json` 与 ZIP 均可见，最好从 Drive 另行下载并校验。确认后再结束 Colab 会话。

在项目目录使用：

```bash
python3 -m artifact_backup /absolute/path/to/downloaded-run.zip
```

已知可信的 adapter 指纹时再加 `--expected-adapter-sha256`。工具不解压文件；拒绝危险路径、符号链接、重复名称、未知文件、缺失核心产物、超大归档、文件哈希不符、基座错配和记录/权重错配。它验证内部一致性；如果归档及清单一起被替换，仅凭内部哈希不能证明来源真实性。

旧 ZIP 没有新 manifest 时不会被冒充为新流程已通过；保留原文件，另行核对其实际权重与原记录的公开指纹。不要把新运行权重配上旧 `evaluation.json` 来绕过检查。

## 中断后的处理

浏览器下载失败时保持会话，先确认已校验的 Drive ZIP，再从 Colab 本地缓存重试下载；不要重新训练。训练已完成而评估失败时，先检查同一 run 的 adapter、baseline 和日志，优先恢复对应产物再续评估。只有日志没有完整权重不能算模型备份成功。

会话丢失但 Drive 完整备份存在时，先下载并离线校验，再用同一基座 revision 加载对应 adapter；本流程不自动切换模型。Drive 未实际保存成功且本机也无权重时，明确记录原权重不可恢复；任何后续复现都建立新的 run、指纹和结果，保留首轮记录。

## 独立扩展评估

`data/eval_v2.jsonl` 包含 28 道新表述/组合题、91 个答案要点别名组：概念 6、流程 8、反例 5、遗漏 5、安全与不确定性 4。题目引用原 33 条知识，**由 agent 编写、未经业务审阅、禁止加入训练**。没有把 RAG 新增的院内示例规则偷偷作为 closed-book 模型必知事实，也不把这些题称为统计独立的外部专业评测集。

Notebook 第 9 节默认 `RUN_EVAL_V2=False`。真实模型、同一运行的 adapter 与已校验备份仍在时才开启；它不训练模型，先临时禁用 adapter 生成基座回答，再恢复 adapter，合计 56 次生成，会额外消耗免费 GPU 配额。保持原生模板、关闭 thinking、greedy、最多 256 新 token。

基座阶段先写 `evaluation_v2_base.json`；完整前后结果另写 `evaluation_v2.json`，标记运行身份、基座 revision、adapter/data 指纹和生成设置，再生成一个新的双副本 ZIP，旧 ZIP 保留。原 `evaluation.json` 与原 8 题指标不变。完整扩展报告已存在时拒绝直接覆盖或重复消耗 GPU。

每题保留完整回答、逐要点同义词组匹配、实际生成 token 数/上限、长度触顶信号、字符四元组重复信号及 `manual_review_status=pending`。字面覆盖只表示某组别名出现；否定句、顺序错误和事实错误也可能命中。重复信号定义为：去除空白后，重叠字符四元组窗口数减不同窗口数，再除窗口数；短于四字为 0。它不能单独判断业务错误。触及输出上限仅提示可能截断，不能直接认定回答不完整。

已有真实回答文件时，可完全离线生成报告：

```bash
python3 -m evaluation_v2 --cases data/eval_v2.jsonl --answers /absolute/path/to/real-answers.jsonl --output /absolute/path/to/new-v2-report.json
```

输入为 JSON 数组或 JSONL；每条必须含 `id`、完整 `answer`、实际 `new_tokens`、实际 `max_new_tokens` 与一致的 `at_token_limit`。不要凭字符数猜 token，也不要编造模型输出。重复、未知或缺失题目 ID 会拒绝生成报告。输出必须使用新文件名；CLI 独占创建文件，拒绝覆盖任何已有文件，并始终禁止写入首轮 `results/evaluation.json`。

最终还须人工逐题核查业务事实、条件/例外、流程先后、关键遗漏、重复和是否完整结束。人工未审的结果不能称为事实准确率；没有真实模型输出时只能说“扩展评估已准备，尚未执行”。
