"""Qwen2.5 HRP 领域微调模型 · Streamlit 在线演示（Streamlit Community Cloud 免费公网）。

用法：
1. Colab 训练完成后把 LoRA 推送到 HF：model.push_to_hub("你的用户名/hrp-qa-qwen-lora")
2. 在 Streamlit Cloud 部署本仓库（Main file 选 streamlit_app.py）
3. 在 Settings → Secrets 添加：
   - BASE_MODEL = Qwen/Qwen2.5-0.5B-Instruct（CPU 免费实例建议 0.5B）
   - LORA_REPO = 你的用户名/hrp-qa-qwen-lora（留空则演示 base 模型）
4. 得到公网地址 https://xxx.streamlit.app
"""
from __future__ import annotations

import os
import streamlit as st

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LORA_REPO = os.getenv("LORA_REPO", "")

st.set_page_config(page_title="HRP 领域问答助手（Qwen 微调）", page_icon="🤖", layout="centered")
st.title("🤖 HRP 医院运营问答助手")
st.caption(f"Base: `{BASE_MODEL}`" + (f" · LoRA: `{LORA_REPO}`" if LORA_REPO else " · 未挂载 LoRA（base 模型演示）"))


@st.cache_resource(show_spinner="首次加载模型（约 1-2 分钟）...")
def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, torch_dtype=torch.float32)
    if LORA_REPO:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_REPO).merge_and_unload()
    return model, tok


model, tok = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

if prompt := st.chat_input("输入问题，例如：什么是HRP系统？"):
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)

    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt")
    with st.chat_message("assistant"):
        with st.spinner("生成中..."):
            out = model.generate(**inputs, max_new_tokens=256, temperature=0.3, top_p=0.9)
            reply = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        st.write(reply)
    st.session_state.history.append(("assistant", reply))
