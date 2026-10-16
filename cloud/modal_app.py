"""Deploy only after verifying free credits and a $0 cash spend limit."""
import os
from pathlib import Path
import modal

from cloud.server import validate_artifacts

artifacts = Path(os.environ.get("HRP_ARTIFACT_DIR", "artifacts")).resolve()
validate_artifacts(artifacts)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "transformers==5.2.0", "peft>=0.18", "accelerate", "fastapi[standard]")
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})
    .add_local_dir(Path(__file__).parent, "/root/cloud")
    .add_local_dir(artifacts, "/root/artifacts")
)
app = modal.App("hrp-qwen35-4b")
model_cache = modal.Volume.from_name("hrp-qwen35-model-cache", create_if_missing=True)


@app.function(image=image, gpu="L4", min_containers=0, max_containers=1,
              scaledown_window=60, timeout=600,
              volumes={"/cache/huggingface": model_cache})
@modal.asgi_app(requires_proxy_auth=True)
def serve():
    from cloud.server import load_api
    return load_api("/root/artifacts")
