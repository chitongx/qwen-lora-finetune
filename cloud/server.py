"""OpenAI-compatible API for the saved Qwen3.5 experiment adapter."""
import json
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-4B"
SERVED_MODEL = "hrp-qwen35-4b"


def validate_artifacts(root):
    root = Path(root)
    config = json.loads((root / "lora_adapter" / "adapter_config.json").read_text())
    evidence = json.loads((root / "evaluation.json").read_text())
    if config["base_model_name_or_path"] != MODEL_ID or evidence["model_id"] != MODEL_ID:
        raise ValueError("adapter 和实验记录必须使用同一 Qwen3.5-4B 基座。")
    if not (root / "lora_adapter" / "adapter_model.safetensors").stat().st_size:
        raise ValueError("adapter 权重为空。")
    return evidence


def create_api(generate, metadata):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    from typing import Literal

    class Message(BaseModel):
        role: Literal["system", "user", "assistant"]
        content: str = Field(min_length=1, max_length=12000)

    class ChatRequest(BaseModel):
        model: str
        messages: list[Message] = Field(min_length=1, max_length=20)
        max_tokens: int = Field(default=256, ge=1, le=256)
        stream: bool = False
        temperature: float = 0

    api = FastAPI()

    @api.get("/health")
    def health():
        return {"model_id": MODEL_ID, "served_model": SERVED_MODEL,
                "model_revision": metadata.get("model_revision"), "adapter_loaded": True}

    @api.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        if request.model != SERVED_MODEL:
            raise HTTPException(400, "Unknown model; this endpoint only serves the HRP adapter.")
        messages = [message.model_dump() for message in request.messages]
        if not request.stream:
            answer = "".join(generate(messages, request.max_tokens))
            return {"object": "chat.completion", "model": SERVED_MODEL, "choices": [
                {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}

        def events():
            try:
                for token in generate(messages, request.max_tokens):
                    data = {"object": "chat.completion.chunk", "model": SERVED_MODEL, "choices": [
                        {"index": 0, "delta": {"content": token}, "finish_reason": None}]}
                    yield "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
                yield "data: [DONE]\n\n"
            except Exception as error:
                yield "data: " + json.dumps({"error": {"message": str(error)}}, ensure_ascii=False) + "\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")
    return api


def load_api(root):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, TextIteratorStreamer
    from threading import Thread

    evidence = validate_artifacts(root)
    adapter = str(Path(root) / "lora_adapter")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=evidence.get("model_revision"), dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter).eval()

    def generate(messages, max_tokens):
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
        if inputs["input_ids"].shape[1] + max_tokens > 4096:
            raise ValueError("本次对话超过演示上下文上限，请开始新对话或缩短问题。")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        errors = []
        def run():
            try:
                with torch.inference_mode():
                    model.generate(**inputs, streamer=streamer, max_new_tokens=max_tokens,
                                   do_sample=False, temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            except Exception as error:
                errors.append(error)
                streamer.on_finalized_text("", stream_end=True)
        worker = Thread(target=run)
        worker.start()
        try:
            yield from streamer
            if errors:
                raise errors[0]
        finally:
            worker.join()
    return create_api(generate, evidence)
