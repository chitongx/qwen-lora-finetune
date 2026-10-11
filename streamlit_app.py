"""Interview evidence viewer with a separately verified live adapter endpoint."""
from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request

import streamlit as st
from cloud_client import chat_cloud, check_cloud

MODEL_ID = "Qwen/Qwen3.5-4B"


def validate_result(result):
    if not isinstance(result, dict):
        raise ValueError("结果 JSON 必须为对象。")
    if result.get("schema_version") != 1 or result.get("model_id") != MODEL_ID:
        raise ValueError("结果文件必须来自本项目的 Qwen3.5-4B 实验。")
    if result["dataset"]["train_count"] != 25 or result["dataset"]["heldout_count"] != 8:
        raise ValueError("本实验固定使用 25 条训练、8 条留出题。")
    for stage in ("base", "tuned"):
        metrics = result[stage]
        rows = metrics["rows"]
        if len(rows) != 8 or len({row["instruction"] for row in rows}) != 8:
            raise ValueError("每个阶段必须包含 8 道不同问题及完整回答。")
        hits, total = 0, 0
        for row in rows:
            keywords, answer = row["keywords"], row["answer"]
            actual = sum(keyword in answer for keyword in keywords)
            if not keywords or actual != row["hit_keywords"] or len(keywords) != row["total_keywords"]:
                raise ValueError("逐题命中计数与保存的回答不一致。")
            hits += actual
            total += len(keywords)
        if hits != metrics["hit_keywords"] or total != metrics["total_keywords"] or abs(metrics["rate"] - hits / total) > 1e-9:
            raise ValueError("汇总命中率与逐题计数不一致。")
    if [(row["instruction"], row["keywords"]) for row in result["base"]["rows"]] != [
        (row["instruction"], row["keywords"]) for row in result["tuned"]["rows"]
    ]:
        raise ValueError("训练前后必须使用相同题目和关键词。")


def chat_ollama(messages):
    payload = json.dumps({"model": "hrp-qa", "messages": messages, "stream": False,
                          "options": {"temperature": 0, "num_predict": 256, "num_ctx": 1024}}).encode()
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["message"]["content"]


st.set_page_config(page_title="HRP 微调对比演示", page_icon="📊", layout="wide")
st.title("HRP 领域微调 · 面试演示")
st.caption(f"{MODEL_ID} · 25 条训练 / 8 条留出评估")
st.info("此页面展示保存的实验回答和字面关键词命中率。8 题小样本指标不等同于事实准确率或生产效果。")

try:
    cloud_endpoint = st.secrets.get("HRP_CLOUD_URL", "")
    cloud_key = st.secrets.get("HRP_CLOUD_API_KEY", "")
except st.errors.StreamlitSecretNotFoundError:
    cloud_endpoint = cloud_key = ""
if cloud_endpoint and cloud_key:
    st.caption("实时新问题：接口已配置；先在下方检查云端模型，通过校验后开放提问。")
else:
    st.warning("实时新问题：未接通云端 GPU。下方是已保存的真实训练记录，不能回答任意新问题。")
st.caption("五分钟演示：先核对真实结果，再查看三单匹配的指标局限与 HRP 的重复失效，最后用已验证的云端模型回答新问题。")

with st.expander("载入另一份实验记录"):
    uploaded = st.file_uploader("载入 Colab 证据包中的 evaluation.json", type=["json"])
saved = Path(__file__).parent / "results" / "evaluation.json"
result = None
try:
    if uploaded is not None:
        result = json.load(uploaded)
    elif saved.exists():
        result = json.loads(saved.read_text(encoding="utf-8"))
    if result is not None:
        validate_result(result)
except (ValueError, KeyError, TypeError) as error:
    st.error(f"无法展示结果：{error}")
    result = None

if result is None:
    st.warning("尚无已载入的真实 GPU 训练结果；不会显示占位命中率。完成 Colab 后上传 evaluation.json，或将它发布为 results/evaluation.json。")
else:
    st.subheader("训练前后证据")
    st.caption("已保存回答 · 非实时生成；所有 8 道留出题均保留完整原文。")
    before, after, delta = st.columns(3)
    before.metric("微调前", f"{result['base']['rate']:.2%}", help=f"{result['base']['hit_keywords']}/{result['base']['total_keywords']} 个关键词")
    after.metric("微调后", f"{result['tuned']['rate']:.2%}", help=f"{result['tuned']['hit_keywords']}/{result['tuned']['total_keywords']} 个关键词")
    delta.metric("变化（百分点）", f"{(result['tuned']['rate'] - result['base']['rate']) * 100:+.2f}")
    if result["tuned"]["rate"] < result["base"]["rate"]:
        st.warning("本轮关键词命中率下降，未证明微调提升；该 adapter 作为实验产物展示。")
    capped = {stage: sum(item["at_token_limit"] for item in result[stage].get("generation", {}).values())
              for stage in ("base", "tuned")}
    st.caption(f"达到输出长度上限：微调前 {capped['base']}/8，微调后 {capped['tuned']}/8；请结合完整回答判断遗漏与重复。")
    questions = [row["instruction"] for row in result["base"]["rows"]]
    if st.session_state.get("evidence_question") not in questions:
        st.session_state.evidence_question = questions[0]
    for column, (label, example) in zip(st.columns(3), [
        ("看完整定义：VMI", "什么是VMI？"),
        ("看指标局限：三单匹配", "什么是三单匹配？"),
        ("看重复失效：HRP", "什么是HRP系统？"),
    ]):
        if column.button(label, disabled=example not in questions):
            st.session_state.evidence_question = example
    question = st.selectbox("查看真实留出题的前后回答", questions, key="evidence_question")
    index = questions.index(question)
    for column, (stage, label) in zip(st.columns(2), [("base", "微调前完整回答"), ("tuned", "微调后完整回答")]):
        with column:
            row = result[stage]["rows"][index]
            st.subheader(label)
            st.write(row["answer"])
            matched = [keyword for keyword in row["keywords"] if keyword in row["answer"]]
            missing = [keyword for keyword in row["keywords"] if keyword not in row["answer"]]
            st.caption(f"关键词 {row['hit_keywords']}/{row['total_keywords']} · 命中：" + ("、".join(matched) or "无"))
            st.caption("未命中：" + ("、".join(missing) or "无"))
            generation = result[stage].get("generation", {}).get(question, {})
            if generation.get("at_token_limit"):
                st.caption("达到 256 token 输出上限；完整性需要人工核对。")
    st.info("人工核对同样重要：三单匹配的“收货单”未命中字面“入库单”；HRP、采购回答存在重复。预算题即使命中关键词，也可能解释错业务规则。")
    with st.expander("运行环境、数据指纹与完整原始记录"):
        st.json(result)
    st.download_button("下载原始结果 JSON", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="evaluation.json", mime="application/json")

if cloud_endpoint and cloud_key:
    st.divider()
    st.subheader("现场新问题 · 云端对话")
    st.caption("检查与提问会唤醒云端 GPU 并消耗免费计算额度；首次启动可能较慢，额度耗尽时停止服务。单次最多输出 256 token。")
    revision = (result or {}).get("model_revision")
    adapter_hash = (result or {}).get("audit", {}).get("adapter_sha256")
    has_identity = bool(revision and adapter_hash)
    if not has_identity:
        st.warning("当前有效训练记录缺少基座版本或 adapter 指纹，不能将实时服务认定为此记录对应的微调模型。")
    if st.button("检查云端模型", disabled=not has_identity):
        try:
            with st.spinner("检查基础模型、训练适配器与运行状态…"):
                st.session_state.cloud_health = check_cloud(cloud_endpoint, cloud_key,
                    model_revision=revision, adapter_sha256=adapter_hash)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, RuntimeError):
            st.session_state.pop("cloud_health", None)
            st.error("云端模型暂未通过检查。可继续查看上方训练记录，或在保留 adapter 的 Colab 会话提问；稍后再检查。")
    health = st.session_state.get("cloud_health", {})
    verified = bool(has_identity and health.get("adapter_loaded") is True
        and health.get("model_id") == MODEL_ID and health.get("served_model") == "hrp-qwen35-4b"
        and health.get("model_revision") == revision and health.get("adapter_sha256") == adapter_hash)
    if not verified:
        st.session_state.pop("cloud_health", None)
    if verified:
        st.success("实时新问题：已验证 Qwen3.5-4B 与训练适配器，可现场提问。")
    else:
        st.caption("实时新问题：待检查；配置接口地址不代表模型已经可用。")
    if "cloud_messages" not in st.session_state:
        st.session_state.cloud_messages = []
    if st.button("开始新对话"):
        st.session_state.cloud_messages = []
    for message in st.session_state.cloud_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("finish_reason") == "length":
                st.warning("此回答达到 256 token 上限，可能未完整结束；请核对业务规则。")
    if prompt := st.chat_input("输入新问题，例如：科室领用耗材需要哪些信息？", disabled=not verified):
        prompt = prompt.strip()
        if not prompt:
            st.warning("请输入非空的 HRP 问题。")
        else:
            conversation = st.session_state.cloud_messages + [{"role": "user", "content": prompt}]
            messages = [{"role": item["role"], "content": item["content"]} for item in conversation]
            with st.chat_message("user"):
                st.write(prompt)
            try:
                generation_info = {}
                with st.chat_message("assistant"), st.spinner("云端模型生成中，首次启动可能较慢…"):
                    answer = st.write_stream(chat_cloud(messages, cloud_endpoint, cloud_key, generation_info=generation_info))
                    if generation_info.get("finish_reason") == "length":
                        st.warning("本次回答达到 256 token 上限，可能未完整结束；请核对业务规则。")
                st.session_state.cloud_messages = conversation + [{"role": "assistant", "content": answer,
                    "finish_reason": generation_info.get("finish_reason")}]
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, RuntimeError):
                st.session_state.pop("cloud_health", None)
                st.error("云端回答未完成，本次问答未保存。可继续查看上方训练记录，或重新检查云端后重试。")
elif os.getenv("HRP_LOCAL_OLLAMA") == "1":
    st.divider()
    st.subheader("本机真实对话 · Ollama hrp-qa")
    st.caption("仅本机服务；请先完成 GGUF 导入。保存的评估回答与此对话独立。")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if prompt := st.chat_input("向本机微调模型提问"):
        messages = st.session_state.messages + [{"role": "user", "content": prompt}]
        with st.chat_message("user"):
            st.write(prompt)
        try:
            with st.chat_message("assistant"), st.spinner("本机模型生成中"):
                answer = chat_ollama(messages)
                st.write(answer)
            st.session_state.messages = messages + [{"role": "assistant", "content": answer}]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
            st.error(f"本机 Ollama 尚未就绪或没有 hrp-qa 模型：{error}")
else:
    st.caption("免费 GPU 暂不可用时，可在保留训练模型的 Colab 会话运行“第 8 节 · 面试现场提问”；它是临时 GPU 演示，不代表持续在线的云端 API。")

with st.expander("演示路线、架构与验收标准"):
    st.markdown("**架构**：25/8 数据划分 → 同基座前后评估 → LoRA adapter 与原始记录 → 轻量网页展示；新问题由独立云端 GPU 推理。")
    st.markdown("**取舍**：4B 适应免费 T4 的资源预算；采用官方建议的非 4bit LoRA。小样本结果下降，不能声称超过 14B 或已优化生产准确率。")
    st.markdown("**验收**：新问题和追问必须实际生成；检查业务事实、遗漏、重复与输出截断。服务不可用时展示已保存记录，并说明当前状态。")
    st.link_button("完整五分钟演示指南", "https://github.com/chitongx/qwen-lora-finetune/blob/main/docs/interview-demo.md")
    st.link_button("打开 Colab Notebook", "https://colab.research.google.com/github/chitongx/qwen-lora-finetune/blob/main/notebooks/qwen_lora_finetune_colab.ipynb")
