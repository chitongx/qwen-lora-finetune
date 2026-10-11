import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None


class AdapterTests(unittest.TestCase):
    def test_cloud_loader_requires_matching_real_adapter_and_evidence(self):
        path = ROOT / "cloud" / "server.py"
        self.assertTrue(path.exists(), "Cloud model loader is missing")
        spec = importlib.util.spec_from_file_location("cloud_server", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "lora_adapter"
            adapter.mkdir()
            with self.assertRaises(FileNotFoundError):
                module.validate_artifacts(root)
            config = adapter / "adapter_config.json"
            config.write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3-14B"}))
            (adapter / "adapter_model.safetensors").write_bytes(b"test-only")
            (root / "evaluation.json").write_text(json.dumps({"model_id": "Qwen/Qwen3.5-4B", "model_revision": "base-commit"}))
            with self.assertRaisesRegex(ValueError, "基座"):
                module.validate_artifacts(root)
            config.write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"}))
            self.assertEqual(module.validate_artifacts(root)["model_revision"], "base-commit")


@unittest.skipUnless(TestClient, "Install FastAPI to run endpoint checks")
class APITests(unittest.TestCase):
    def test_endpoint_streams_full_conversation_and_rejects_wrong_model(self):
        path = ROOT / "cloud" / "server.py"
        self.assertTrue(path.exists(), "Cloud inference API is missing")
        spec = importlib.util.spec_from_file_location("cloud_server", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        seen = []
        def generate(messages, max_tokens):
            seen.append((messages, max_tokens))
            yield "医院"
            yield "资源"
        client = TestClient(module.create_api(generate, {"model_id": "Qwen/Qwen3.5-4B"}))
        messages = [{"role": "user", "content": "什么是HRP？"},
                    {"role": "assistant", "content": "医院资源规划"},
                    {"role": "user", "content": "然后呢？"}]
        response = client.post("/v1/chat/completions", json={"model": "hrp-qwen35-4b", "messages": messages, "stream": True})
        self.assertEqual(response.status_code, 200)
        self.assertIn('"content": "医院"', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
        self.assertEqual(seen, [(messages, 256)])
        self.assertEqual(client.post("/v1/chat/completions", json={"model": "wrong", "messages": messages}).status_code, 400)
        self.assertEqual(client.post("/v1/chat/completions", json={"model": "hrp-qwen35-4b", "messages": messages, "max_tokens": 9999}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
