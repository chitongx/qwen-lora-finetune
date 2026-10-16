import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class ArtifactBackupTests(unittest.TestCase):
    def module(self):
        path = ROOT / "artifact_backup.py"
        self.assertTrue(path.exists(), "Training needs a durable run and verified bundle helper")
        spec = importlib.util.spec_from_file_location("artifact_backup", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def populate(self, artifacts):
        adapter = artifacts / "lora_adapter"
        adapter.mkdir()
        weights = b"test-only-adapter"
        (adapter / "adapter_model.safetensors").write_bytes(weights)
        (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"}))
        (adapter / "tokenizer_config.json").write_text("{}")
        (adapter / "tokenizer.json").write_text("{}")
        (artifacts / "evaluation.json").write_text(json.dumps({
            "model_id": "Qwen/Qwen3.5-4B", "model_revision": "base-revision",
            "audit": {"adapter_sha256": hashlib.sha256(weights).hexdigest()}}))

    def test_prepare_run_requires_existing_writable_storage_and_never_reuses_run(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "持久"):
                module.prepare_run(storage / "unmounted")
            first = module.prepare_run(storage)
            second = module.prepare_run(storage)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            module.assert_backup_ready(first)
            with patch.object(Path, "write_bytes", side_effect=OSError("quota exhausted")):
                with self.assertRaisesRegex(RuntimeError, "持久"):
                    module.assert_backup_ready(first)

    def test_export_verifies_matching_weights_in_drive_and_local_copies(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "drive"
            storage.mkdir()
            artifacts = module.prepare_run(storage)
            self.populate(artifacts)
            local = module.build_verified_bundle(artifacts, Path(tmp) / "cache")
            drive = artifacts.parent / local.name
            self.assertTrue(local.is_file())
            self.assertTrue(drive.is_file())
            self.assertEqual(local.read_bytes(), drive.read_bytes())
            report = module.verify_bundle(local)
            self.assertEqual(report["run_id"], artifacts.parent.name)
            expected = hashlib.sha256(b"test-only-adapter").hexdigest()
            self.assertEqual(report["adapter_sha256"], expected)
            module.verify_bundle(drive, expected_adapter_sha256=expected)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                module.verify_bundle(local, expected_adapter_sha256="wrong")

    def test_export_rejects_adapter_from_a_different_run(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            artifacts = module.prepare_run(storage)
            self.populate(artifacts)
            (artifacts / "lora_adapter" / "adapter_model.safetensors").write_bytes(b"another-run")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                module.build_verified_bundle(artifacts, storage / "cache")

    def test_empty_weights_are_not_a_verified_adapter_even_when_hash_matches(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            artifacts = module.prepare_run(storage)
            self.populate(artifacts)
            (artifacts / "lora_adapter" / "adapter_model.safetensors").write_bytes(b"")
            evidence = json.loads((artifacts / "evaluation.json").read_text())
            evidence["audit"]["adapter_sha256"] = hashlib.sha256(b"").hexdigest()
            (artifacts / "evaluation.json").write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "为空"):
                module.build_verified_bundle(artifacts, storage / "cache")

    def test_zip_rejects_unsafe_paths_links_duplicates_and_unexpected_files(self):
        module = self.module()
        entries = ["../escape", "/absolute", "lora_adapter/../../escape", "C:\\escape", "credentials.json"]
        with tempfile.TemporaryDirectory() as tmp:
            for index, entry in enumerate(entries):
                archive = Path(tmp) / f"bad-{index}.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(entry, b"test-only")
                with self.subTest(entry=entry), self.assertRaises(ValueError):
                    module.verify_bundle(archive)
            archive = Path(tmp) / "link.zip"
            link = zipfile.ZipInfo("lora_adapter/adapter_model.safetensors")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(link, "/outside")
            with self.assertRaisesRegex(ValueError, "链接"):
                module.verify_bundle(archive)
            archive = Path(tmp) / "duplicate.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("evaluation.json", "{}")
                with self.assertWarns(UserWarning):
                    bundle.writestr("evaluation.json", "{}")
            with self.assertRaisesRegex(ValueError, "重复"):
                module.verify_bundle(archive)

    def test_zip_rejects_modified_file_bytes_even_with_valid_crc(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            artifacts = module.prepare_run(storage)
            self.populate(artifacts)
            local = module.build_verified_bundle(artifacts, storage / "cache")
            bad = storage / "tampered.zip"
            with zipfile.ZipFile(local) as original, zipfile.ZipFile(bad, "w") as bundle:
                for name in original.namelist():
                    bundle.writestr(name, b"changed" if name.endswith("adapter_model.safetensors") else original.read(name))
            with self.assertRaisesRegex(ValueError, "SHA256|大小"):
                module.verify_bundle(bad)


if __name__ == "__main__":
    unittest.main()
