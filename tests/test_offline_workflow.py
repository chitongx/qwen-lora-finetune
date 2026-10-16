"""Offline regressions: execute notebook data/evaluation code without a GPU or model download."""
import ast
from collections import Counter
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"


def notebook():
    return json.loads((ROOT / "notebooks/qwen_lora_finetune_colab.ipynb").read_text())


def sources():
    return ["".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == "code"]


def run_data():
    from artifact_backup import assert_backup_ready, prepare_run
    src = next(s for s in sources() if "data = seed" in s or "# DATA" in s)
    ns = {}
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        # An uploaded seed is optional: the embedded project dataset must work standalone.
        import os
        old = os.getcwd()
        try:
            os.chdir(tmp)
            ns.update(artifacts=prepare_run(Path(tmp)), assert_backup_ready=assert_backup_ready)
            exec(src, ns)
        finally:
            os.chdir(old)
    if "eval_set" not in ns:
        for src in sources():
            for node in ast.parse(src.replace("!pip", "# !pip")).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "eval_set" for t in node.targets):
                    ns["eval_set"] = ast.literal_eval(node.value)
    return ns


class WorkflowTests(unittest.TestCase):
    def test_drive_authorization_failure_stops_before_dependency_install(self):
        source = sources()[0]
        self.assertIn("drive.mount", source, "Persistent output must be authorized before setup or training")
        mount = SimpleNamespace(mount=lambda path: (_ for _ in ()).throw(RuntimeError("human authorization required")))
        with patch.dict(sys.modules, {"google.colab": SimpleNamespace(drive=mount)}), \
             patch("subprocess.run", return_value=SimpleNamespace(stdout="Tesla T4, 15360")), \
             patch("subprocess.check_call") as install:
            with self.assertRaisesRegex(RuntimeError, "human authorization required"):
                exec(source, {"__name__": "__notebook__"})
        install.assert_not_called()

    def test_training_cell_checks_persistent_write_access_before_model_mutation(self):
        source = next(src for src in sources() if "training_result = trainer.train()" in src)
        first = ast.parse(source).body[0]
        self.assertIsInstance(first, ast.Expr, "Training must stop before imports/model mutation if backup is not ready")
        self.assertEqual(getattr(first.value.func, "id", None), "assert_backup_ready")
        from artifact_backup import assert_backup_ready
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "持久"):
                exec(compile(ast.Module(body=[first], type_ignores=[]), "<training-preflight>", "exec"),
                     {"assert_backup_ready": assert_backup_ready, "artifacts": Path(tmp)})

    def test_expanded_evaluation_is_opt_in_and_requires_preserved_model_and_backup(self):
        cells = [src for src in sources() if "def run_evaluation_v2(" in src]
        self.assertEqual(len(cells), 1, "Independent evaluation needs a separately runnable, disabled-by-default cell")
        source = cells[0]
        self.assertIn("RUN_EVAL_V2 = False", source)
        self.assertNotIn("trainer.train", source)
        self.assertNotIn('artifacts / "evaluation.json").write_text', source)
        namespace = {"__name__": "__notebook__"}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, namespace)
        self.assertEqual(len(namespace["eval_v2_cases"]), 28)
        with self.assertRaisesRegex(RuntimeError, "会话|备份"):
            namespace["run_evaluation_v2"]()

    def test_expanded_evaluation_keeps_original_record_and_backs_up_new_full_answers(self):
        cells = [src for src in sources() if "def run_evaluation_v2(" in src]
        self.assertEqual(len(cells), 1, "Expanded evaluation has not been integrated")
        from artifact_backup import (assert_backup_ready, build_verified_bundle,
                                     prepare_run, sha256_file, verify_bundle)
        class Model:
            peft_config = {"default": object()}
            config = SimpleNamespace(_commit_hash="base-revision")
            disabled = False
            @contextlib.contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            artifacts = prepare_run(storage)
            adapter = artifacts / "lora_adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"test-only-weights")
            (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": MODEL_ID}))
            (adapter / "tokenizer_config.json").write_text("{}")
            (adapter / "tokenizer.json").write_text("{}")
            original = json.dumps({"model_id": MODEL_ID, "model_revision": "base-revision", "audit": {
                "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors")}})
            (artifacts / "evaluation.json").write_text(original)
            archive = build_verified_bundle(artifacts, storage / "cache")
            model = Model()
            log, calls = {}, []
            def generate(current, tokenizer, question, max_new=256):
                calls.append((current.disabled, question))
                log[question] = {"new_tokens": 3, "at_token_limit": False}
                return "完整基座回答" if current.disabled else "完整微调回答"
            namespace = {"__name__": "__notebook__", "model": model, "tokenizer": object(),
                "generate": generate, "generation_log": log, "FastLanguageModel": SimpleNamespace(for_inference=lambda model: None),
                "artifacts": artifacts, "archive": archive, "backup_cache": storage / "cache",
                "assert_backup_ready": assert_backup_ready, "build_verified_bundle": build_verified_bundle,
                "sha256_file": sha256_file, "verify_bundle": verify_bundle, "MODEL_ID": MODEL_ID,
                "data_hash": lambda rows: hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(cells[0], namespace)
                report = namespace["run_evaluation_v2"]()
            self.assertEqual((artifacts / "evaluation.json").read_text(), original)
            self.assertEqual(report["base"]["summary"]["cases"], 28)
            self.assertEqual(report["tuned"]["summary"]["cases"], 28)
            self.assertEqual(len(calls), 56)
            self.assertTrue(all(disabled for disabled, _ in calls[:28]))
            self.assertTrue(all(not disabled for disabled, _ in calls[28:]))
            self.assertEqual(report["tuned"]["rows"][0]["answer"], "完整微调回答")
            self.assertTrue((artifacts / "evaluation_v2_base.json").exists())
            self.assertTrue((artifacts / "evaluation_v2.json").exists())
            saved_zip = namespace["archive"]
            self.assertNotEqual(saved_zip, archive)
            with zipfile.ZipFile(saved_zip) as bundle:
                saved = json.loads(bundle.read("evaluation_v2.json"))
            self.assertEqual(saved["tuned"]["rows"][0]["answer"], "完整微调回答")
            verify_bundle(artifacts.parent / saved_zip.name)

    def interview_function(self):
        cells = [src for src in sources() if "def interview_compare(" in src]
        self.assertEqual(len(cells), 1, "Notebook needs a separately runnable live interview cell")
        tree = ast.parse(cells[0])
        funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "interview_compare"]
        self.assertNotIn("trainer.train", cells[0])
        self.assertNotIn("write_text", cells[0], "Live Q&A must not overwrite the original evaluation")
        return funcs

    def test_interview_cell_compares_loaded_base_and_adapter_without_retraining(self):
        funcs = self.interview_function()
        class AdapterModel:
            peft_config = {"default": object()}
            disabled = False
            @contextlib.contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False
        model = AdapterModel()
        calls = []
        log = {}
        def generate(current, tokenizer, question, max_new=256):
            calls.append((current.disabled, question, max_new))
            log[question] = {"new_tokens": 3, "at_token_limit": False}
            return "基座真实输出" if current.disabled else "adapter 真实输出"
        ns = {"model": model, "tokenizer": SimpleNamespace(apply_chat_template=lambda *args, **kw: [1] * 10),
              "generate": generate, "generation_log": log, "max_seq_length": 1024,
              "FastLanguageModel": SimpleNamespace(for_inference=lambda current: None)}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "<interview>", "exec"), ns)
        result = ns["interview_compare"]("  科室领用耗材需要什么信息？  ")
        self.assertEqual(result["base_answer"], "基座真实输出")
        self.assertEqual(result["tuned_answer"], "adapter 真实输出")
        self.assertEqual(calls, [(True, "科室领用耗材需要什么信息？", 256),
                                 (False, "科室领用耗材需要什么信息？", 256)])
        self.assertFalse(model.disabled, "Adapter must be restored after the base comparison")
        self.assertEqual(result["tuned_generation"]["new_tokens"], 3)

    def test_interview_cell_rejects_lost_runtime_and_oversized_question(self):
        funcs = self.interview_function()
        ns = {}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "<interview>", "exec"), ns)
        with self.assertRaisesRegex(RuntimeError, "会话"):
            ns["interview_compare"]("新问题")
        ns.update(model=SimpleNamespace(peft_config={"default": object()}),
                  tokenizer=SimpleNamespace(apply_chat_template=lambda *args, **kw: [1] * 900),
                  generate=lambda *args: self.fail("Must reject overlength before generation"),
                  generation_log={}, max_seq_length=1024,
                  FastLanguageModel=SimpleNamespace(for_inference=lambda current: None))
        with self.assertRaisesRegex(ValueError, "上下文"):
            ns["interview_compare"]("很长的问题")

    def test_text_evaluation_accepts_processor_or_plain_tokenizer(self):
        src = next(s for s in sources() if "def generate(" in s)
        assignments = [n for n in ast.parse(src).body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "tokenizer" for t in n.targets)]
        self.assertTrue(assignments, "Qwen3.5 loader returns a multimodal processor; text evaluation must unwrap its tokenizer")
        text_tokenizer = object()
        for loaded in [SimpleNamespace(tokenizer=text_tokenizer), text_tokenizer]:
            ns = {"tokenizer": loaded}
            exec(compile(ast.Module(body=assignments, type_ignores=[]), "<text-tokenizer>", "exec"), ns)
            self.assertIs(ns["tokenizer"], text_tokenizer)

    def test_standalone_notebook_uses_25_training_rows_from_33_seed_rows(self):
        ns = run_data()
        self.assertEqual(len(ns["data"]), 25, "Notebook must use the real dataset and exclude 8 held-out questions")
        self.assertEqual(len(ns["seed"]), 33)
        self.assertEqual(len(ns["eval_set"]), 8)

    def test_heldout_questions_are_not_training_questions(self):
        ns = run_data()
        train_questions = {x["instruction"] for x in ns["data"]}
        eval_questions = {x.get("instruction", x.get("q")) for x in ns["eval_set"]}
        self.assertFalse(train_questions & eval_questions, "Evaluation must not reuse training questions")
        self.assertEqual(len(train_questions | eval_questions), 33)

    def test_comparison_uses_one_model_and_runs_baseline_before_training(self):
        src = "\n".join(sources())
        self.assertNotIn("Qwen/Qwen2.5", src, "Baseline must use the same base as the trained model")
        self.assertIn(MODEL_ID, src)
        self.assertLess(src.index("base_metrics = evaluate("), src.index("trainer.train("))
        self.assertGreater(src.index("tuned_metrics = evaluate("), src.index("trainer.train("))
        self.assertNotIn("del model", src, "Keep the trained model alive for GGUF export")

    def test_keyword_evaluation_keeps_full_answers_and_counts(self):
        src = next(s for s in sources() if "def evaluate(" in s)
        funcs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "evaluate"]
        full_answer = "医院资源规划 " + "完整回答" * 60 + " 业务财务一体化"
        ns = {"eval_set": [{"instruction": "问题", "keywords": ["医院资源规划", "未命中"], "output": "标准", "source": "seed:1", "q": "问题", "kw": ["医院资源规划", "未命中"]}],
              "generate": lambda *args, **kwargs: full_answer}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "<evaluation>", "exec"), ns)
        result = ns["evaluate"](object(), object())
        self.assertIsInstance(result, dict, "Evaluation must preserve machine-readable evidence, not truncated tuples")
        self.assertEqual(result["hit_keywords"], 1)
        self.assertEqual(result["total_keywords"], 2)
        self.assertEqual(result["rate"], 0.5)
        self.assertEqual(result["rows"][0]["answer"], full_answer)

    def test_generator_reproduces_committed_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "scripts").mkdir()
            (target / "notebooks").mkdir()
            shutil.copytree(ROOT / "data", target / "data")
            shutil.copy(ROOT / "scripts/gen_notebook.py", target / "scripts/gen_notebook.py")
            shutil.copy(ROOT / "artifact_backup.py", target / "artifact_backup.py")
            shutil.copy(ROOT / "evaluation_v2.py", target / "evaluation_v2.py")
            completed = subprocess.run([sys.executable, str(target / "scripts/gen_notebook.py")], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            actual = json.loads((target / "notebooks/qwen_lora_finetune_colab.ipynb").read_text())
            self.assertEqual(actual, notebook())

    def test_streamlit_does_not_load_transformer_weights_in_free_cloud(self):
        tree = ast.parse((ROOT / "streamlit_app.py").read_text())
        imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        imports |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertFalse(imports & {"torch", "transformers", "peft"}, "Cloud app must not allocate a multi-GB local model")

    def test_notebook_preserves_artifacts_before_optional_gguf(self):
        src = "\n".join(sources())
        self.assertIn("do_sample=False", src)
        self.assertIn("apply_chat_template", src)
        self.assertIn("num_train_epochs=3", src)
        self.assertNotIn("/no_think", src)
        self.assertLess(src.index("archive = build_verified_bundle("), src.index("model.save_pretrained_gguf("))
        self.assertIn("RUN_GGUF_EXPORT = False", src)
        self.assertIn("max_memory_allocated", src)
        self.assertIn("sha256", src)

    def test_notebook_exports_fingerprint_of_actual_saved_adapter_in_json_and_zip(self):
        from artifact_backup import build_verified_bundle, prepare_run, verify_bundle
        source = next(src for src in sources() if "archive = build_verified_bundle" in src)
        nodes = ast.parse(source).body
        start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets))
        stop = next(i for i, node in enumerate(nodes) if isinstance(node, ast.ImportFrom)
                    and node.module == "google.colab")
        code = compile(ast.Module(body=nodes[start:stop], type_ignores=[]), "<artifact-export>", "exec")
        import os
        fingerprints = []
        for weights in [b"saved-adapter-weights-A", b"saved-adapter-weights-B"]:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                artifacts = prepare_run(root)
                adapter = artifacts / "lora_adapter"
                adapter.mkdir(parents=True)
                (adapter / "adapter_model.safetensors").write_bytes(weights)
                (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": MODEL_ID}))
                (adapter / "tokenizer_config.json").write_text("{}")
                (adapter / "tokenizer.json").write_text("{}")
                ns = {"artifacts": artifacts, "adapter_dir": adapter, "json": json, "hashlib": hashlib,
                    "shutil": shutil, "MODEL_ID": MODEL_ID, "backup_cache": root / "cache",
                    "build_verified_bundle": build_verified_bundle, "verify_bundle": verify_bundle,
                    "model": SimpleNamespace(config=SimpleNamespace(_commit_hash="base-revision")),
                    "datetime": datetime, "timezone": timezone, "dataset_manifest": {},
                    "base_metrics": {"rate": 0.72}, "tuned_metrics": {"rate": 0.68},
                    "training_result": SimpleNamespace(metrics={}), "SEED": 3407,
                    "max_seq_length": 1024, "training_precision": {}, "parameter_info": {},
                    "environment": {}, "gpu_info": {}, "memory": {}}
                previous = os.getcwd()
                try:
                    os.chdir(root)
                    exec(code, ns)
                finally:
                    os.chdir(previous)
                expected = hashlib.sha256(weights).hexdigest()
                result = json.loads((artifacts / "evaluation.json").read_text())
                self.assertIn("audit", result, "Fresh notebook must export adapter identity without a manual post-cell")
                self.assertEqual(result["audit"]["adapter_sha256"], expected)
                with zipfile.ZipFile(ns["archive"]) as bundle:
                    archived = json.loads(bundle.read("evaluation.json"))
                    self.assertEqual(archived["audit"]["adapter_sha256"], expected)
                    self.assertEqual(hashlib.sha256(bundle.read("lora_adapter/adapter_model.safetensors")).hexdigest(), expected)
                fingerprints.append(expected)
        self.assertNotEqual(*fingerprints)

    def test_no_gpu_fails_before_install_or_model_download(self):
        src = sources()[0]
        self.assertIn("subprocess.check_call(", src, "Dependency installation must stop on failure")
        self.assertIn("nvidia-smi", src)
        self.assertLess(src.index("nvidia-smi"), src.index("subprocess.check_call("))
        self.assertIn("raise RuntimeError", src)
        self.assertIn("transformers==5.2.0", src)
        self.assertIn("trl==0.22.2", src)

    def test_qwen35_uses_unquantized_lora_and_disables_thinking_consistently(self):
        parsed = ast.parse("\n".join(sources()))
        load = next(node for node in ast.walk(parsed) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained")
        quantized = next(kw.value for kw in load.keywords if kw.arg == "load_in_4bit")
        self.assertIs(ast.literal_eval(quantized), False)
        templates = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "apply_chat_template"]
        self.assertEqual(len(templates), 3, "Training, evaluation and live interview templates disable thinking")
        for call in templates:
            thinking = next(kw.value for kw in call.keywords if kw.arg == "enable_thinking")
            self.assertIs(ast.literal_eval(thinking), False)

    def test_deployment_only_exposes_loopback_and_keeps_chat_history(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        self.assertIn("127.0.0.1:11434:11434", compose)
        app = (ROOT / "streamlit_app.py").read_text()
        self.assertIn("http://127.0.0.1:11434/api/chat", app)
        template = (ROOT / "Modelfile").read_text()
        self.assertIn("range .Messages", template)
        self.assertNotIn(".Prompt", template)

    def test_published_result_validation_rejects_model_mismatch_and_fake_counts(self):
        tree = ast.parse((ROOT / "streamlit_app.py").read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_result"]
        self.assertEqual(len(funcs), 1, "Published evidence needs validation before displaying rates")
        ns = {"MODEL_ID": MODEL_ID}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "<app-validation>", "exec"), ns)
        with self.assertRaises(ValueError):
            ns["validate_result"]({"model_id": "Qwen/Qwen2.5-0.5B-Instruct"})
        rows = [{"instruction": f"问题{i}", "keywords": ["领域"], "answer": "领域", "hit_keywords": 1, "total_keywords": 1} for i in range(8)]
        result = {"schema_version": 1, "model_id": MODEL_ID, "base": {"hit_keywords": 8, "total_keywords": 8, "rate": 1.0, "rows": rows}, "tuned": {"hit_keywords": 8, "total_keywords": 8, "rate": 1.0, "rows": rows}, "dataset": {"train_count": 25, "heldout_count": 8}}
        self.assertIsNone(ns["validate_result"](result))
        result["tuned"]["hit_keywords"] = 7
        with self.assertRaises(ValueError):
            ns["validate_result"](result)

    def test_published_result_validation_rejects_non_object_json(self):
        tree = ast.parse((ROOT / "streamlit_app.py").read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_result"]
        ns = {"MODEL_ID": MODEL_ID}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "<app-validation>", "exec"), ns)
        with self.assertRaises(ValueError):
            ns["validate_result"]([])

    def test_local_chat_preserves_full_messages_and_uses_trained_model(self):
        import urllib.request
        tree = ast.parse((ROOT / "streamlit_app.py").read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "chat_ollama"]
        ns = {"json": json, "urllib": __import__("urllib")}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "<local-chat>", "exec"), ns)
        messages = [{"role": "user", "content": "HRP是什么？"}, {"role": "assistant", "content": "医院资源规划"}, {"role": "user", "content": "它管理什么？"}]
        with patch.object(urllib.request, "urlopen", return_value=io.BytesIO(b'{"message":{"content":"real endpoint response"}}')) as call:
            self.assertEqual(ns["chat_ollama"](messages), "real endpoint response")
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["model"], "hrp-qa")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["temperature"], 0)

    def test_t4_requests_fp16_without_emulated_bf16(self):
        # Unsloth may override the requested precision; record its actual flags separately.
        fake_torch = SimpleNamespace(bfloat16="bf16", float16="fp16",
            cuda=SimpleNamespace(is_bf16_supported=lambda including_emulation=True: including_emulation))
        parsed = ast.parse("\n".join(sources()))
        load = next(node for node in ast.walk(parsed) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained")
        dtype = next(kw.value for kw in load.keywords if kw.arg == "dtype")
        value = eval(compile(ast.Expression(dtype), "<model-dtype>", "eval"), {"torch": fake_torch})
        self.assertEqual(value, "fp16", "T4's BF16 emulation must not select model BF16")
        amp = next(node.value for node in parsed.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "use_bf16" for t in node.targets))
        value = eval(compile(ast.Expression(amp), "<training-amp>", "eval"), {"torch": fake_torch})
        self.assertFalse(value, "T4 must not request emulated BF16 training")

    def test_evidence_records_precision_after_trainer_override(self):
        parsed = ast.parse("\n".join(sources()))
        assignment = next((node for node in parsed.body if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "training_precision"
                                   for t in node.targets)), None)
        self.assertIsNotNone(assignment, "Evidence must record actual trainer flags after Unsloth overrides")
        namespace = {
            "Counter": Counter,
            "trainer": SimpleNamespace(args=SimpleNamespace(fp16=False, bf16=False)),
            "model": SimpleNamespace(parameters=lambda: [SimpleNamespace(dtype="float32"),
                SimpleNamespace(dtype="float32"), SimpleNamespace(dtype="float16")]),
        }
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "<precision>", "exec"), namespace)
        self.assertEqual(namespace["training_precision"], {
            "fp16": False, "bf16": False,
            "parameter_tensor_dtypes": {"float32": 2, "float16": 1},
        })


if __name__ == "__main__":
    unittest.main()
