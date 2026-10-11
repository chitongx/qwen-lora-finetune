import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_RESULT = json.loads((ROOT / "results" / "evaluation.json").read_text())
HEALTH = {"model_id": "Qwen/Qwen3.5-4B", "served_model": "hrp-qwen35-4b", "adapter_loaded": True,
          "model_revision": PUBLISHED_RESULT["model_revision"],
          "adapter_sha256": PUBLISHED_RESULT["audit"]["adapter_sha256"]}


@unittest.skipUnless(AppTest, "Install Streamlit to run UI integration checks")
class CloudUITests(unittest.TestCase):
    def test_interview_cases_show_original_answers_and_missing_keywords(self):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
        self.assertTrue(any("实时新问题：未接通" in item.value for item in at.warning))
        self.assertTrue(any("已保存回答" in item.value for item in at.caption))
        self.assertIn("看指标局限：三单匹配", [item.label for item in at.button])
        next(item for item in at.button if item.label == "看指标局限：三单匹配").click().run()
        self.assertEqual(at.selectbox[0].value, "什么是三单匹配？")
        self.assertTrue(any("未命中：入库单" in item.value for item in at.caption))
        self.assertTrue(any("收货单（GRN）" in item.value for item in at.markdown))
        next(item for item in at.button if item.label == "看重复失效：HRP").click().run()
        self.assertEqual(at.selectbox[0].value, "什么是HRP系统？")
        self.assertTrue(any(item.value.count("成本核算") > 2 for item in at.markdown))

    def test_configured_cloud_requires_identity_check_and_can_clear_history(self):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        at.secrets["HRP_CLOUD_URL"] = "https://example.modal.run"
        at.secrets["HRP_CLOUD_API_KEY"] = "test-only"
        at.run()
        self.assertTrue(at.chat_input[0].disabled, "Configured URL alone is not a verified live adapter")
        with patch("cloud_client.check_cloud", return_value=HEALTH):
            next(item for item in at.button if item.label == "检查云端模型").click().run()
        self.assertFalse(at.chat_input[0].disabled)
        with patch("cloud_client.chat_cloud", return_value=iter(["真实响应"])):
            at.chat_input[0].set_value("什么是VMI？").run()
        self.assertEqual(len(at.session_state["cloud_messages"]), 2)
        next(item for item in at.button if item.label == "开始新对话").click().run()
        self.assertEqual(at.session_state["cloud_messages"], [])

    def test_cloud_error_does_not_save_partial_answer_and_keeps_evidence_available(self):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        at.secrets["HRP_CLOUD_URL"] = "https://example.modal.run"
        at.secrets["HRP_CLOUD_API_KEY"] = "test-only"
        at.run()
        self.assertTrue(any(item.label == "检查云端模型" for item in at.button))
        with patch("cloud_client.check_cloud", return_value=HEALTH):
            next(item for item in at.button if item.label == "检查云端模型").click().run()
        def interrupted(*args, **kwargs):
            yield "半截回答"
            raise RuntimeError("test stream interrupted")
        with patch("cloud_client.chat_cloud", side_effect=interrupted):
            at.chat_input[0].set_value("问题").run()
        self.assertEqual(at.session_state["cloud_messages"], [])
        self.assertTrue(any("训练记录" in item.value for item in at.error))
        self.assertEqual([item.value for item in at.metric], ["72.00%", "68.00%", "-4.00"])

    def test_live_length_cap_is_visible_without_editing_answer(self):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        at.secrets["HRP_CLOUD_URL"] = "https://example.modal.run"
        at.secrets["HRP_CLOUD_API_KEY"] = "test-only"
        at.run()
        with patch("cloud_client.check_cloud", return_value=HEALTH):
            next(item for item in at.button if item.label == "检查云端模型").click().run()
        def capped(messages, endpoint, key, *, generation_info):
            yield "原始回答"
            generation_info["finish_reason"] = "length"
        with patch("cloud_client.chat_cloud", side_effect=capped):
            at.chat_input[0].set_value("新问题").run()
        self.assertTrue(any("本次回答达到 256 token" in item.value for item in at.warning))
        self.assertEqual(at.session_state["cloud_messages"][-1]["content"], "原始回答")
        at.run()
        self.assertTrue(any("达到 256 token" in item.value for item in at.warning),
                        "Output limit must remain visible when the answer becomes history")
        with patch("cloud_client.chat_cloud", return_value=iter(["后续回答"])) as call:
            at.chat_input[0].set_value("请继续解释").run()
        self.assertTrue(all(set(message) == {"role", "content"} for message in call.call_args.args[0]))
        self.assertEqual(at.session_state["cloud_messages"][1]["finish_reason"], "length")

    def test_whitespace_question_does_not_wake_gpu(self):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        at.secrets["HRP_CLOUD_URL"] = "https://example.modal.run"
        at.secrets["HRP_CLOUD_API_KEY"] = "test-only"
        at.run()
        with patch("cloud_client.check_cloud", return_value=HEALTH):
            next(item for item in at.button if item.label == "检查云端模型").click().run()
        with patch("cloud_client.chat_cloud") as call:
            at.chat_input[0].set_value("   ").run()
        call.assert_not_called()
        self.assertEqual(at.session_state["cloud_messages"], [])
        self.assertTrue(any("非空" in item.value for item in at.warning))

    def test_evidence_page_runs_without_cloud_credentials(self):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.chat_input), 0)

    def test_published_run_discloses_regression_and_truncated_answers(self):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
        self.assertEqual([item.value for item in at.metric], ["72.00%", "68.00%", "-4.00"])
        self.assertEqual(at.metric[2].label, "变化（百分点）")
        self.assertTrue(any("下降" in item.value for item in at.warning))
        self.assertTrue(any("8/8" in item.value and "2/8" in item.value for item in at.caption))

    def test_cloud_chat_preserves_full_followup_history(self):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        at.secrets["HRP_CLOUD_URL"] = "https://example.modal.run"
        at.secrets["HRP_CLOUD_API_KEY"] = "test-only"
        at.run()
        self.assertTrue(any(item.label == "检查云端模型" for item in at.button))
        with patch("cloud_client.check_cloud", return_value=HEALTH):
            next(item for item in at.button if item.label == "检查云端模型").click().run()
        with patch("cloud_client.chat_cloud", return_value=iter(["医院资源规划"])) as call:
            at.chat_input[0].set_value("什么是HRP？").run()
            self.assertEqual(call.call_args.args[0], [{"role": "user", "content": "什么是HRP？"}])
        with patch("cloud_client.chat_cloud", return_value=iter(["人财物"])) as call:
            at.chat_input[0].set_value("它管什么？").run()
            self.assertEqual(call.call_args.args[0], [
                {"role": "user", "content": "什么是HRP？"},
                {"role": "assistant", "content": "医院资源规划"},
                {"role": "user", "content": "它管什么？"},
            ])
        self.assertEqual(len(at.exception), 0)

    def test_changing_record_invalidates_previously_verified_adapter(self):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        at.secrets["HRP_CLOUD_URL"] = "https://example.modal.run"
        at.secrets["HRP_CLOUD_API_KEY"] = "test-only"
        at.run()
        with patch("cloud_client.check_cloud", return_value=HEALTH):
            next(item for item in at.button if item.label == "检查云端模型").click().run()
        self.assertFalse(at.chat_input[0].disabled)
        changed = json.loads(json.dumps(PUBLISHED_RESULT))
        changed["audit"]["adapter_sha256"] = "another-adapter-hash"
        with patch("streamlit.file_uploader", return_value=io.StringIO(json.dumps(changed))):
            at.run()
        self.assertTrue(at.chat_input[0].disabled)
        self.assertFalse(any("已验证" in item.value for item in at.success))

    def test_record_without_adapter_fingerprint_cannot_verify_live_identity(self):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        at.secrets["HRP_CLOUD_URL"] = "https://example.modal.run"
        at.secrets["HRP_CLOUD_API_KEY"] = "test-only"
        unaudited = json.loads(json.dumps(PUBLISHED_RESULT))
        unaudited.pop("audit")
        with patch("streamlit.file_uploader", return_value=io.StringIO(json.dumps(unaudited))):
            at.run()
        self.assertTrue(at.chat_input[0].disabled)
        self.assertTrue(next(item for item in at.button if item.label == "检查云端模型").disabled)
        self.assertTrue(any("指纹" in item.value for item in at.warning))


if __name__ == "__main__":
    unittest.main()
