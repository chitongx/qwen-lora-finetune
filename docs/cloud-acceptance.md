# 两个项目的云端接入和验收

目前公网RAG可以实时检索，LoRA页面可以展示首轮真实实验。下面步骤完成并实际验证前，不将公网生成、权重持久备份或在线推理标为完成。免费方案以账户实际适用额度为准，现金支出要求为零。

## 用户一次准备的账号事项

1. 打开 [Modal](https://modal.com/login?next=%2Fapps)，自行完成登录或注册、条款与验证码，使用Starter。在Usage & Billing确认可用计算额度，并将 **spend limit（抵扣额度后的现金支出上限）设为0**；它与抵扣前的usage limit不同。若要求购买套餐或付款，先停止。可用额度、上限以账户页面为准，见[官方预算说明](https://modal.com/docs/guide/budgets)。不要使用从首个token起收费的Shared Endpoints。
2. 用同一个Google账号登录Colab和Drive。用户亲自完成Notebook发起的Drive挂载授权；程序只读写 `MyDrive/HRP_LoRA_Backups/` 中本次独立run，不索取Token或Cookie。首个持久目录读写检查失败时不会开始安装或训练。
3. 确认可管理 [RAG应用](https://rag-knowledge-base01.streamlit.app/) 和 [LoRA应用](https://qwen35-hrp-lora.streamlit.app/) 的Settings → Secrets。推理地址与凭据确定后由用户直接在平台输入，不发聊天、不写仓库或本地认证文件。需要创建访问凭据或授予部署权限的步骤交给用户本人确认和操作。

不要求额外注册Hugging Face或购买套餐。GitHub代码发布沿用已有chitongx访问权限。账号未就绪不妨碍离线数据、评估与程序检查，但不能完成真正的云端模型验收。

## 先取得可恢复的实际训练产物

- 在新版Notebook第一节完成Drive授权和持久写入检查；每次运行自动创建独立目录，不覆盖首轮72%→68%记录。
- 若上次权重确已丢失，在存储就绪后重现实验并单独保存新run；不要把新权重搭配旧JSON冒充首轮产物。
- 第5节保存adapter、tokenizer、完整评估和逐文件manifest，生成并核对Drive与Colab本地ZIP。用户在Drive网页另行查看该目录，确认真实远端文件存在；触发浏览器下载后，还需在Mac确认ZIP并运行 `python artifact_backup.py ZIP路径`。
- 云端解包后先核对同一run的基座revision和adapter SHA256，再启动模型。目录或ZIP校验不等于模型已经实际加载。

具体操作与恢复要求见[备份与扩展评估](backup-and-evaluation.md)。

## 接入两个公网页面

经授权的部署进程使用 `cloud/modal_app.py`；加载真实Qwen3.5-4B和该run的adapter。只在账户额度和零现金上限确认后运行。部署得到的HTTPS根地址为下表中的“接口根地址”；本文不提供虚构可用地址。

| 应用 | 配置项 | 含义 |
|---|---|---|
| LoRA | `HRP_CLOUD_URL` | 实际接口根地址，不带 `/v1` |
| LoRA | `HRP_CLOUD_API_KEY` | 由本人保管的Modal推理访问凭据 |
| RAG | `LLM_API_BASE` | 同一实际接口根地址加 `/v1` |
| RAG | `LLM_MODEL` | `hrp-qwen35-4b` |
| RAG | `LLM_API_KEY` | 由本人保管的该接口访问凭据 |

接口保留Proxy鉴权，未授权请求不应进入GPU生成。LoRA页同时需要上传本次run对应的 `evaluation.json`，再检查云端模型；不能拿页面默认首轮记录去校验新run的权重。RAG是否采用当前adapter仍取决于真实生成质量，不因两个项目共用服务就视为adapter优于基座。

## 最后实际验收

1. 断开原训练会话后，从持久副本重新加载；LoRA页检查实际model/revision/adapter SHA256，并问“科室领用耗材需要哪些信息？”，再追问“这些信息分别有什么用？”。保留完整回答与是否触达上限的记录。
2. RAG问“药品和耗材入库要核对什么？”，观察实际逐段输出，逐项核对来源编号对应的原文。单次无错不等于所有引用均可靠。
3. RAG问资料不包含的问题，例如“医院食堂本周菜单是什么？”。实际检查是否说明资料不足，不能把最近邻候选当成答案，也不能只凭出现拒答短语就认定整篇回答正确。
4. 记录实际日期、接口模型身份、资料和题集版本、完整回答与错误；运行扩展生成评估后，逐题人工核对事实、遗漏和引用支撑关系。未运行的项目保留待验收状态。

平台额度耗尽、GPU不可用或接口报错时，保留真实实验记录作为演示备用，不静默切换为付费服务或其他模型。
