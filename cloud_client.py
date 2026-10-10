"""Stream the trained cloud model without loading weights in Streamlit."""
import json
import urllib.request


def chat_cloud(messages, endpoint, api_key):
    payload = {"model": "hrp-qwen35-4b", "messages": messages,
               "stream": True, "temperature": 0, "max_tokens": 256}
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                return
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"]["message"])
            content = event["choices"][0]["delta"].get("content")
            if content:
                yield content
    raise RuntimeError("云端回答中断，请重试；本次回答没有保存为完整对话。")
