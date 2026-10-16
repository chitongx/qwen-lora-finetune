"""Standard-library run storage and internally consistent adapter backup checks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
from uuid import uuid4
import zipfile

MODEL_ID = "Qwen/Qwen3.5-4B"
ROOT_FILES = {"backup_ready.json", "train.jsonl", "heldout.jsonl", "dataset_manifest.json",
              "baseline.json", "evaluation.json", "training_log.json", "evaluation_v2.json", "evaluation_v2_base.json"}
ADAPTER_FILES = {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json",
                 "tokenizer.json", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
                 "vocab.json", "merges.txt", "generation_config.json", "README.md"}
REQUIRED_FILES = {"backup_ready.json", "evaluation.json", "lora_adapter/adapter_config.json",
                  "lora_adapter/adapter_model.safetensors", "lora_adapter/tokenizer_config.json",
                  "lora_adapter/tokenizer.json"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_backup_ready(artifacts):
    artifacts = Path(artifacts)
    try:
        ready = json.loads((artifacts / "backup_ready.json").read_text())
        if ready["run_id"] != artifacts.parent.name:
            raise ValueError("run identity mismatch")
        probe = artifacts / ".write_probe"
        probe.write_bytes(b"hrp-persistent-output-check")
        if probe.read_bytes() != b"hrp-persistent-output-check":
            raise OSError("read-back mismatch")
        probe.unlink()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("持久输出目录未通过写入回读检查；停止，不开始训练。") from error


def prepare_run(storage):
    storage = Path(storage)
    if not storage.is_dir():
        raise RuntimeError("持久存储未就绪；请先由本人完成 Google Drive 挂载授权。")
    run_id = "qwen35-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    artifacts = storage / run_id / "artifacts"
    try:
        artifacts.mkdir(parents=True, exist_ok=False)
        (artifacts / "backup_ready.json").write_text(json.dumps({
            "run_id": run_id, "write_checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Filesystem write/read check; verify the saved run separately in Drive before ending the session."}, indent=2))
        assert_backup_ready(artifacts)
    except OSError as error:
        raise RuntimeError("持久目录创建或写入失败；停止，不开始训练。") from error
    return artifacts


def allowed_file(name):
    return name in ROOT_FILES or name.startswith("lora_adapter/") and name.split("/", 1)[1] in ADAPTER_FILES


def validate_identity(evidence, config, actual_hash):
    if actual_hash == hashlib.sha256(b"").hexdigest():
        raise ValueError("adapter 权重为空，不能视作已验证备份。")
    if evidence.get("model_id") != MODEL_ID or config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError("备份基座必须为 Qwen3.5-4B。")
    if not evidence.get("model_revision"):
        raise ValueError("备份缺少基础模型 revision。")
    if evidence.get("audit", {}).get("adapter_sha256") != actual_hash:
        raise ValueError("adapter SHA256 与同一运行的 evaluation.json 不一致。")


def verify_bundle(archive, expected_adapter_sha256=None):
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 64 or sum(item.file_size for item in entries) > 512 * 1024**2:
            raise ValueError("ZIP 超出本项目 adapter 备份大小限制。")
        names = []
        for item in entries:
            name = item.filename
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("ZIP 不允许符号链接。")
            if "\\" in name or ":" in name or name.startswith("/") or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/")):
                raise ValueError("ZIP 含危险路径。")
            if item.is_dir():
                if name != "lora_adapter/":
                    raise ValueError("ZIP 含未知目录。")
                continue
            if name in names:
                raise ValueError("ZIP 含重复文件名。")
            if name != "manifest.json" and not allowed_file(name):
                raise ValueError("ZIP 含本项目备份清单之外的文件。")
            names.append(name)
        if not REQUIRED_FILES | {"manifest.json"} <= set(names):
            raise ValueError("ZIP 缺少 adapter、tokenizer、结果或 manifest。")
        manifest = json.loads(bundle.read("manifest.json"))
        if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(names) - {"manifest.json"}:
            raise ValueError("manifest 文件清单不完整。")
        for name, expected in manifest["files"].items():
            content = bundle.read(name)
            if len(content) != expected["size"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
                raise ValueError("文件大小或 SHA256 与 manifest 不一致。")
        evidence = json.loads(bundle.read("evaluation.json"))
        config = json.loads(bundle.read("lora_adapter/adapter_config.json"))
        actual_hash = manifest["files"]["lora_adapter/adapter_model.safetensors"]["sha256"]
        validate_identity(evidence, config, actual_hash)
        ready = json.loads(bundle.read("backup_ready.json"))
        if manifest.get("run_id") != ready.get("run_id") or manifest.get("model_revision") != evidence["model_revision"]:
            raise ValueError("备份运行身份或基座 revision 不一致。")
        if manifest.get("adapter_sha256") != actual_hash or expected_adapter_sha256 and expected_adapter_sha256 != actual_hash:
            raise ValueError("adapter SHA256 不符合预期。")
    return {"run_id": manifest["run_id"], "model_id": MODEL_ID, "model_revision": evidence["model_revision"],
            "adapter_sha256": actual_hash, "zip_sha256": sha256_file(archive), "file_count": len(names)}


def build_verified_bundle(artifacts, local_cache):
    artifacts = Path(artifacts)
    assert_backup_ready(artifacts)
    files = {}
    for path in sorted(artifacts.rglob("*")):
        if path.is_symlink():
            raise ValueError("备份不允许符号链接。")
        if not path.is_file():
            continue
        name = path.relative_to(artifacts).as_posix()
        if name == "manifest.json":
            continue
        if not allowed_file(name):
            raise ValueError("拒绝打包未知文件；备份仅包含训练与评估产物。")
        files[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    if not REQUIRED_FILES <= set(files):
        raise ValueError("adapter/tokenizer/评估记录尚未保存完整，不能生成已验证备份。")
    evidence = json.loads((artifacts / "evaluation.json").read_text())
    config = json.loads((artifacts / "lora_adapter" / "adapter_config.json").read_text())
    adapter_hash = files["lora_adapter/adapter_model.safetensors"]["sha256"]
    validate_identity(evidence, config, adapter_hash)
    manifest = {"schema_version": 1, "run_id": artifacts.parent.name, "model_id": MODEL_ID,
                "model_revision": evidence["model_revision"], "adapter_sha256": adapter_hash, "files": files}
    (artifacts / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    cache = Path(local_cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (artifacts.parent.name + "-" + uuid4().hex[:8] + ".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(set(files) | {"manifest.json"}):
            bundle.write(artifacts / name, name)
    local_report = verify_bundle(archive)
    drive_copy = artifacts.parent / archive.name
    shutil.copyfile(archive, drive_copy)
    if verify_bundle(drive_copy) != local_report:
        raise ValueError("持久 ZIP 与本地 ZIP 校验不一致；请保留会话并检查存储。")
    return archive


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Read-only check of a downloaded HRP run ZIP; never extracts files.")
    parser.add_argument("archive")
    parser.add_argument("--expected-adapter-sha256")
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.archive, args.expected_adapter_sha256), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
