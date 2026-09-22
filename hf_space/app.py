"""HuggingFace Spaces 公网演示：HRP 领域微调模型 Chat 界面（Gradio）。

部署（SDK=Gradio）：
1. 在 Colab notebook 第 5 步把微调后的 LoRA 推送到 HF：model.push_to_hub("你的用户名/hrp-qa-qwen-lora")
2. 在 HF 新建 Space，SDK 选 Gradio，上传本目录 app.py + requirements.txt
3. 在 Space Settings -> Variables 添加：
   - BASE_MODEL = Qwen/Qwen2.5-0.5B-Instruct（CPU 免费实例建议 0.5B；7B 需 GPU）
   - LORA_REPO = 你的用户名/hrp-qa-qwen-lora（留空则加载纯 base 模型）
4. 完成后获得公网地址 https://<用户名>-<space名>.hf.space，可直接放简历
"""
import os
import gradio as gr

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LORA_REPO = os.getenv("LORA_REPO", "")

_model, _tok = None, None


def load_model():
    global _model, _tok
    if _model is not None:
        return _model, _tok
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, torch_dtype=torch.float32)
    if LORA_REPO:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_REPO)
        model = model.merge_and_unload()
    _model, _tok = model, tok
    return model, tok


def chat(message, history):
    model, tok = load_model()
    msgs = [{"role": "user", "content": message}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.3, top_p=0.9)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


demo = gr.ChatInterface(
    fn=chat,
    title="HRP 医院运营问答助手（Qwen 微调）",
    description="基于 Qwen2.5 + QLoRA 微调的领域模型。"
                f"Base: {BASE_MODEL}" + (f" | LoRA: {LORA_REPO}" if LORA_REPO else "（未挂载 LoRA）"),
)

if __name__ == "__main__":
    demo.launch()
