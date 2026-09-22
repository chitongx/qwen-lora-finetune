#!/usr/bin/env bash
# Qwen 微调模型 Ollama 一键部署 + 验收
# 前置：把 Colab 导出的 .gguf 文件放到 ./models/ 目录
set -euo pipefail
cd "$(dirname "$0")"

GGUF=$(ls models/*.gguf 2>/dev/null | head -1 || true)
if [ -z "${GGUF}" ]; then
  echo "❌ 未找到 models/*.gguf"
  echo "请先完成："
  echo "  1) 在 Colab 跑通 qwen_lora_finetune_colab.ipynb（第 7 步导出 GGUF）"
  echo "  2) 把 model-Q4_K_M.gguf（约 2GB）放到 ./models/ 目录"
  exit 1
fi
echo "找到模型文件: ${GGUF}"

echo "[1/3] 启动 Ollama"
docker compose up -d

echo "[2/3] 等待 Ollama 就绪"
for i in $(seq 1 30); do
  if curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then break; fi
  sleep 2
done

echo "[3/3] 创建模型 hrp-qa（首次导入约 1-3 分钟）"
NAME=$(basename "${GGUF}")
cat > Modelfile.generated <<EOF
FROM /root/.ollama/models/${NAME}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.3
EOF
docker compose cp Modelfile.generated ollama:/Modelfile.generated
docker compose exec -T ollama ollama create hrp-qa -f /Modelfile.generated

echo "✅ 部署完成，验收："
echo 'curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '\''{"model":"hrp-qa","messages":[{"role":"user","content":"什么是HRP系统？"}]}'\'''
echo ""
echo "接入 RAG 项目（rag-knowledge-base/.env）："
echo "  LLM_API_BASE=http://localhost:11434/v1"
echo "  LLM_MODEL=hrp-qa"
