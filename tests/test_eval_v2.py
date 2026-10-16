"""Independent-question preparation and offline evaluation, without a model or GPU."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EvaluationV2Tests(unittest.TestCase):
    def evaluator(self):
        self.assertIsNotNone(importlib.util.find_spec("evaluation_v2"),
                             "The embeddable, standard-library evaluation library is missing")
        return importlib.import_module("evaluation_v2")

    def sample_case(self):
        return {"id": "example", "category": "process", "question": "先核对哪些单据？",
                "source_seed_indices": [11],
                "provenance": {"author": "agent", "status": "unreviewed",
                               "source_file": "data/hrp_qa_seed.jsonl",
                               "source_indexing": "zero_based", "not_for_training": True},
                "manual_review_required": True,
                "rubric_points": [{"id": "receipt", "description": "核对收货记录",
                                   "aliases": ["入库单", "收货单"]},
                                  {"id": "invoice", "description": "核对发票",
                                   "aliases": ["发票"]}]}

    def test_dataset_has_28_new_questions_with_unreviewed_source_and_alias_rubrics(self):
        path = ROOT / "data/eval_v2.jsonl"
        self.assertTrue(path.exists(), "The expanded evaluation questions have not been prepared")
        cases = self.evaluator().load_cases(path)
        seed = [json.loads(line) for line in (ROOT / "data/hrp_qa_seed.jsonl").read_text().splitlines()]
        old = json.loads((ROOT / "results/evaluation.json").read_text())
        original_questions = {row["instruction"] for row in seed}
        original_questions.update(row["instruction"] for row in old["base"]["rows"])
        self.assertEqual(len(cases), 28)
        self.assertEqual(len({row["id"] for row in cases}), 28)
        self.assertEqual(len({row["question"] for row in cases}), 28)
        self.assertFalse(original_questions & {row["question"] for row in cases})
        self.assertEqual({row["category"] for row in cases},
                         {"concepts", "process", "counterexamples", "omissions", "safety_uncertainty"})
        source_indices = set()
        for row in cases:
            self.assertIs(row["manual_review_required"], True)
            self.assertEqual(row["provenance"]["author"], "agent")
            self.assertEqual(row["provenance"]["status"], "unreviewed")
            self.assertEqual(row["provenance"]["source_file"], "data/hrp_qa_seed.jsonl")
            self.assertEqual(row["provenance"]["source_indexing"], "zero_based")
            self.assertIs(row["provenance"]["not_for_training"], True)
            self.assertTrue(row["source_seed_indices"])
            source_indices.update(row["source_seed_indices"])
            self.assertTrue(all(type(i) is int and 0 <= i < 33 for i in row["source_seed_indices"]))
            self.assertGreaterEqual(len(row["rubric_points"]), 2)
            self.assertTrue(all(point["description"] and point["aliases"]
                                and all(isinstance(alias, str) and alias.strip() for alias in point["aliases"])
                                for point in row["rubric_points"]))
        self.assertEqual(source_indices, set(range(33)), "New questions must stay traceable to all existing seed knowledge")

    def test_alias_groups_keep_full_answer_and_pending_manual_review(self):
        answer = "先核对收货单。" + "审计记录完整。" * 80
        calls = []
        def generate(question):
            calls.append(question)
            return {"answer": answer, "new_tokens": 100, "max_new_tokens": 256,
                    "at_token_limit": False}
        report = self.evaluator().evaluate_cases([self.sample_case()], generate)
        row = report["rows"][0]
        self.assertEqual(calls, ["先核对哪些单据？"])
        self.assertEqual(row["answer"], answer)
        self.assertEqual(row["rubric_matches"][0]["matched_aliases"], ["收货单"])
        self.assertTrue(row["rubric_matches"][0]["matched"])
        self.assertFalse(row["rubric_matches"][1]["matched"])
        self.assertEqual(report["summary"]["literal_rubric_match_rate"], 0.5)
        self.assertEqual(report["summary"]["manual_review_pending_cases"], 1)
        self.assertEqual(row["manual_review_status"], "pending")
        self.assertEqual(row["generation"]["new_tokens"], 100)
        self.assertIn("not factual accuracy", report["metric_definitions"]["literal_rubric_match_rate"])
        self.assertNotIn("accuracy", report["summary"])

    def test_repeated_four_gram_excess_ignores_whitespace_and_short_answers(self):
        evaluator = self.evaluator()
        self.assertEqual(evaluator.repeated_four_gram_excess_ratio("医 院\n"), 0.0)
        self.assertEqual(evaluator.repeated_four_gram_excess_ratio("医院物资财务"), 0.0)
        # Four identical overlapping 4-grams, of which three are excess occurrences.
        self.assertEqual(evaluator.repeated_four_gram_excess_ratio("啊 啊啊\n啊啊啊啊"), 0.75)

    def test_token_limit_is_reported_as_a_signal_not_proven_truncation(self):
        report = self.evaluator().evaluate_cases([self.sample_case()], lambda question: {
            "answer": "采购订单、收货单、发票", "new_tokens": 256,
            "max_new_tokens": 256, "at_token_limit": True})
        self.assertEqual(report["rows"][0]["truncation_status"], "token_limit_reached")
        self.assertEqual(report["summary"]["at_token_limit_cases"], 1)
        self.assertIn("not proof", report["metric_definitions"]["at_token_limit_cases"])
        with self.assertRaisesRegex(ValueError, "at_token_limit"):
            self.evaluator().evaluate_cases([self.sample_case()], lambda question: {
                "answer": "完整文本", "new_tokens": 256, "max_new_tokens": 256,
                "at_token_limit": False})

    def test_generation_metadata_is_required_without_inventing_token_counts(self):
        with self.assertRaisesRegex(ValueError, "new_tokens"):
            self.evaluator().evaluate_cases([self.sample_case()], lambda question: {"answer": "单据"})

    def test_literal_matches_do_not_resolve_negation_or_complete_manual_review(self):
        report = self.evaluator().evaluate_cases([self.sample_case()], lambda question: {
            "answer": "不需要收货单，也不用发票。", "new_tokens": 12,
            "max_new_tokens": 256, "at_token_limit": False})
        self.assertEqual(report["summary"]["literal_rubric_match_rate"], 1.0,
                         "The documented proxy detects strings, never factual truth")
        self.assertEqual(report["rows"][0]["manual_review_status"], "pending")
        self.assertIn("Negation", report["metric_definitions"]["literal_rubric_match_rate"])

    def test_invalid_token_counts_are_rejected(self):
        for new_tokens, limit in [(True, 256), (-1, 256), (257, 256), (1, 0)]:
            with self.subTest(new_tokens=new_tokens, limit=limit):
                with self.assertRaisesRegex(ValueError, "tokens"):
                    self.evaluator().evaluate_cases([self.sample_case()], lambda question: {
                        "answer": "单据", "new_tokens": new_tokens,
                        "max_new_tokens": limit, "at_token_limit": False})

    def test_duplicate_case_ids_and_empty_aliases_are_rejected(self):
        evaluator = self.evaluator()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text("\n".join(json.dumps(self.sample_case(), ensure_ascii=False) for _ in range(2)))
            with self.assertRaisesRegex(ValueError, "duplicate.*id"):
                evaluator.load_cases(path)
            case = self.sample_case()
            case["rubric_points"][0]["aliases"] = [""]
            path.write_text(json.dumps(case, ensure_ascii=False))
            with self.assertRaisesRegex(ValueError, "aliases"):
                evaluator.load_cases(path)

    def run_cli(self, answer_rows, use_jsonl=False):
        self.evaluator()
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / "cases.jsonl"
            answers = Path(tmp) / ("answers.jsonl" if use_jsonl else "answers.json")
            output = Path(tmp) / "report.json"
            cases.write_text(json.dumps(self.sample_case(), ensure_ascii=False))
            answers.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in answer_rows)
                               if use_jsonl else json.dumps(answer_rows, ensure_ascii=False))
            result = subprocess.run([sys.executable, str(ROOT / "evaluation_v2.py"),
                                     "--cases", str(cases), "--answers", str(answers),
                                     "--output", str(output)], capture_output=True, text=True)
            report = json.loads(output.read_text()) if output.exists() else None
            return result, report

    def test_offline_cli_reads_json_and_jsonl_without_a_model(self):
        answer = {"id": "example", "answer": "收货单和发票", "new_tokens": 8,
                  "max_new_tokens": 256, "at_token_limit": False}
        for use_jsonl in [False, True]:
            with self.subTest(use_jsonl=use_jsonl):
                result, report = self.run_cli([answer], use_jsonl=use_jsonl)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(report["summary"]["literal_rubric_match_rate"], 1.0)
                self.assertTrue(report["prepared_data_unreviewed"])

    def test_offline_cli_rejects_duplicate_missing_and_unknown_answer_ids(self):
        answer = {"id": "example", "answer": "收货单", "new_tokens": 8,
                  "max_new_tokens": 256, "at_token_limit": False}
        for records, message in [([answer, answer], "duplicate"), ([], "missing"),
                                 ([dict(answer, id="unrelated")], "unknown")]:
            with self.subTest(message=message):
                result, report = self.run_cli(records)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertIsNone(report, "Invalid inputs must not produce a partial success report")

    def write_cli_inputs(self, root):
        cases, answers = root / "cases.jsonl", root / "answers.json"
        cases.write_text(json.dumps(self.sample_case(), ensure_ascii=False))
        answers.write_text(json.dumps([{"id": "example", "answer": "收货单和发票",
            "new_tokens": 8, "max_new_tokens": 256, "at_token_limit": False}], ensure_ascii=False))
        return ["--cases", str(cases), "--answers", str(answers)]

    def test_offline_cli_preserves_existing_output_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arguments = self.write_cli_inputs(root)
            output = root / "existing-report.json"
            original = b"existing evidence must survive\x00\xff"
            output.write_bytes(original)
            result = subprocess.run([sys.executable, str(ROOT / "evaluation_v2.py"), *arguments,
                "--output", str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_offline_cli_always_reserves_original_result_path(self):
        for exists in [False, True]:
            with self.subTest(exists=exists), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                script = root / "evaluation_v2.py"
                script.write_bytes((ROOT / "evaluation_v2.py").read_bytes())
                arguments = self.write_cli_inputs(root)
                output = root / "results" / "evaluation.json"
                output.parent.mkdir()
                original = b"original first-run evidence"
                if exists:
                    output.write_bytes(original)
                result = subprocess.run([sys.executable, str(script), *arguments,
                    "--output", str(output)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("reserved", result.stderr)
                self.assertEqual(output.read_bytes() if output.exists() else None, original if exists else None)

    def test_offline_cli_exclusive_create_preserves_a_competing_writer(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from unittest.mock import patch
        evaluator = self.evaluator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arguments = self.write_cli_inputs(root)
            output = root / "race-report.json"
            original = b"record created by competing writer"
            real_open = Path.open
            def racing_open(path, mode="r", *args, **kwargs):
                if path == output and mode in {"w", "x"} and not output.exists():
                    with real_open(output, "wb") as stream:
                        stream.write(original)
                return real_open(path, mode, *args, **kwargs)
            stderr = StringIO()
            with patch.object(sys, "argv", ["evaluation_v2", *arguments, "--output", str(output)]), \
                 patch.object(Path, "open", racing_open), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exit_error:
                    evaluator.main()
            self.assertEqual(exit_error.exception.code, 2)
            self.assertIn("File exists", stderr.getvalue())
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
