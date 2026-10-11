"""Stream the trained cloud model without loading weights in Streamlit."""
import json
import urllib.request


def check_cloud(endpoint, api_key, *, model_revision=None, adapter_sha256=None):
    request = urllib.request.Request(endpoint.rstrip("/") + "/health",
        headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        metadata = json.load(response)
    expected = {"model_id": "Qwen/Qwen3.5-4B", "served_model": "hrp-qwen35-4b", "adapter_loaded": True}
    if model_revision:
        expected["model_revision"] = model_revision
    if adapter_sha256:
        expected["adapter_sha256"] = adapter_sha256
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("云端模型或适配器与本次实验不一致，未通过身份校验。")
    return metadata


def chat_cloud(messages, endpoint, api_key, *, generation_info=None):
    payload = {"model": "hrp-qwen35-4b", "messages": messages,
               "stream": True, "temperature": 0, "max_tokens": 256}
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    has_content = False
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                if not has_content:
                    raise RuntimeError("云端返回空回答，请重试。")
                return
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"]["message"])
            choice = event["choices"][0]
            if generation_info is not None and choice.get("finish_reason"):
                generation_info["finish_reason"] = choice["finish_reason"]
            content = choice["delta"].get("content")
            if content:
                has_content = has_content or bool(content.strip())
                yield content
    raise RuntimeError("云端回答中断，请重试；本次回答没有保存为完整对话。")
