import importlib.util
import hashlib
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
    def test_finish_reason_uses_output_tokens_and_end_token(self):
        spec = importlib.util.spec_from_file_location("cloud_server", ROOT / "cloud" / "server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "generation_finish_reason"))
        self.assertEqual(module.generation_finish_reason([4, 5, 6], 3, 99), "length")
        self.assertEqual(module.generation_finish_reason([4, 5, 99], 3, [98, 99]), "stop")
        self.assertEqual(module.generation_finish_reason([4, 99], 3, 99), "stop")

    def test_cloud_loader_rejects_weights_different_from_audited_experiment(self):
        spec = importlib.util.spec_from_file_location("cloud_server", ROOT / "cloud" / "server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "lora_adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"}))
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"actual-test-weights")
            evidence = {"model_id": "Qwen/Qwen3.5-4B", "audit": {
                "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}}
            (root / "evaluation.json").write_text(json.dumps(evidence))
            self.assertEqual(module.validate_artifacts(root)["audit"], evidence["audit"])
            weights.write_bytes(b"different-test-weights")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                module.validate_artifacts(root)

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
    def test_length_capped_generation_is_not_reported_as_natural_stop(self):
        spec = importlib.util.spec_from_file_location("cloud_server", ROOT / "cloud" / "server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def generate(messages, max_tokens):
            yield "到上限的原文"
            return "length"
        client = TestClient(module.create_api(generate, {}))
        request = {"model": "hrp-qwen35-4b", "messages": [{"role": "user", "content": "问题"}]}
        response = client.post("/v1/chat/completions", json=request)
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")
        response = client.post("/v1/chat/completions", json=dict(request, stream=True))
        self.assertIn('"finish_reason": "length"', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    def test_health_reports_actual_adapter_fingerprint(self):
        spec = importlib.util.spec_from_file_location("cloud_server", ROOT / "cloud" / "server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        client = TestClient(module.create_api(lambda *args: iter([]), {
            "model_revision": "base-revision", "adapter_sha256": "actual-weight-hash"}))
        health = client.get("/health").json()
        self.assertEqual(health.get("adapter_sha256"), "actual-weight-hash")
        self.assertTrue(health["adapter_loaded"])

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
