import io
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]


class CloudChatTests(unittest.TestCase):
    def load_client(self):
        self.assertTrue((ROOT / "cloud_client.py").exists(), "Cloud conversation client is missing")
        spec = importlib.util.spec_from_file_location("cloud_client", ROOT / "cloud_client.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_stream_sends_complete_history_and_yields_actual_tokens(self):
        client = self.load_client()
        messages = [{"role": "user", "content": "HRP是什么？"},
                    {"role": "assistant", "content": "医院资源规划"},
                    {"role": "user", "content": "它管理什么？"}]
        output = 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: {"choices":[{"delta":{"content":"人财"}}]}\n\ndata: {"choices":[{"delta":{"content":"物"}}]}\n\ndata: [DONE]\n\n'
        with patch("urllib.request.urlopen", return_value=io.BytesIO(output.encode())) as call:
            self.assertEqual(list(client.chat_cloud(messages, "https://example.modal.run", "test-only")), ["人财", "物"])
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.modal.run/v1/chat/completions")
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["model"], "hrp-qwen35-4b")
        self.assertTrue(payload["stream"])
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only")

    def test_incomplete_stream_is_reported_instead_of_saved_as_complete(self):
        client = self.load_client()
        output = 'data: {"choices":[{"delta":{"content":"一部分"}}]}\n\n'
        with patch("urllib.request.urlopen", return_value=io.BytesIO(output.encode())):
            with self.assertRaisesRegex(RuntimeError, "中断"):
                list(client.chat_cloud([], "https://example.modal.run", "test-only"))

    def test_server_error_is_reported_instead_of_empty_answer(self):
        client = self.load_client()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b'data: {"error":{"message":"quota exhausted"}}\n\n')):
            with self.assertRaisesRegex(RuntimeError, "quota exhausted"):
                list(client.chat_cloud([], "https://example.modal.run", "test-only"))


if __name__ == "__main__":
    unittest.main()
