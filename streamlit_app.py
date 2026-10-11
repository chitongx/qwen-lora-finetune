"""Lightweight experiment evidence viewer; local chat uses an existing Ollama model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request

import streamlit as st
from cloud_client import chat_cloud

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


st.set_page_config(page_title="HRP LoRA 实验记录", page_icon="📊", layout="centered")
st.title("HRP 领域 LoRA 实验")
st.caption(f"{MODEL_ID} · 25 条训练 / 8 条留出评估")
st.info("此页面展示保存的实验回答和字面关键词命中率。8 题小样本指标不等同于事实准确率或生产效果。")

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
    before, after, delta = st.columns(3)
    before.metric("微调前", f"{result['base']['rate']:.2%}", help=f"{result['base']['hit_keywords']}/{result['base']['total_keywords']} 个关键词")
    after.metric("微调后", f"{result['tuned']['rate']:.2%}", help=f"{result['tuned']['hit_keywords']}/{result['tuned']['total_keywords']} 个关键词")
    delta.metric("变化", f"{(result['tuned']['rate'] - result['base']['rate']) * 100:+.2f} 个百分点")
    questions = [row["instruction"] for row in result["base"]["rows"]]
    question = st.selectbox("查看真实留出题的前后回答", questions)
    index = questions.index(question)
    for stage, label in [("base", "微调前完整回答"), ("tuned", "微调后完整回答")]:
        row = result[stage]["rows"][index]
        st.subheader(label)
        st.write(row["answer"])
        st.caption(f"关键词 {row['hit_keywords']}/{row['total_keywords']}：" + "、".join(row["keywords"]))
    with st.expander("运行环境、数据指纹与完整原始记录"):
        st.json(result)
    st.download_button("下载原始结果 JSON", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="evaluation.json", mime="application/json")

try:
    cloud_endpoint = st.secrets.get("HRP_CLOUD_URL", "")
    cloud_key = st.secrets.get("HRP_CLOUD_API_KEY", "")
except st.errors.StreamlitSecretNotFoundError:
    cloud_endpoint = cloud_key = ""
if cloud_endpoint and cloud_key:
    st.divider()
    st.subheader("HRP 云端对话")
    st.caption("Qwen3.5-4B + 本次训练 adapter；首次回答可能需要等待云端启动，免费额度耗尽时会停止服务。")
    if "cloud_messages" not in st.session_state:
        st.session_state.cloud_messages = []
    for message in st.session_state.cloud_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if prompt := st.chat_input("输入 HRP 问题"):
        messages = st.session_state.cloud_messages + [{"role": "user", "content": prompt}]
        with st.chat_message("user"):
            st.write(prompt)
        try:
            with st.chat_message("assistant"), st.spinner("云端模型生成中，首次启动可能较慢…"):
                answer = st.write_stream(chat_cloud(messages, cloud_endpoint, cloud_key))
            st.session_state.cloud_messages = messages + [{"role": "assistant", "content": answer}]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, RuntimeError) as error:
            st.error(f"云端暂不可用，请稍后重试：{error}")
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
    st.caption("训练后的新问题对话将接入免费额度内的云端 GPU 服务；目前此页面仅承载实验记录，云端接口尚在接入。")
