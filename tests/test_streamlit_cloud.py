import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(AppTest, "Install Streamlit to run UI integration checks")
class CloudUITests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
