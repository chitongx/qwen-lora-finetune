#!/usr/bin/env bash
# Import the user's trained GGUF, then save a real Ollama inference response.
set -euo pipefail
cd "$(dirname "$0")"
shopt -s nullglob
ggufs=(models/*.gguf)
if [ "${#ggufs[@]}" -ne 1 ]; then
  echo "请先完成 Colab 可选 GGUF 导出，并将唯一一个实际生成的 .gguf 放入 models/。"
  echo "发现 ${#ggufs[@]} 个文件；不会猜测或下载其他基础模型代替。"
  exit 1
fi
mode="${1:-native}"
case "$mode" in
  native)
    command -v ollama >/dev/null || { echo "请先安装并启动原生 Ollama：https://ollama.com/download"; exit 1; }
    model_path="./${ggufs[0]}"
    ;;
  docker)
    model_path="/gguf/$(basename "${ggufs[0]}")"
    ;;
  *) echo "用法：./deploy.sh [native|docker]"; exit 1 ;;
esac
printf 'FROM "%s"\n' "$model_path" > Modelfile.generated
sed '1,2d' Modelfile >> Modelfile.generated
if [ "$mode" = docker ]; then
  docker compose up -d
fi
ready=0
for ((attempt=1; attempt<=30; attempt++)); do
  if curl --fail --silent --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" -ne 1 ]; then
  echo "Ollama 未就绪；原生模式请先启动 Ollama app 或在另一终端运行 ollama serve。"
  exit 1
fi
if [ "$mode" = native ]; then
  ollama create hrp-qa -f Modelfile.generated
else
  docker compose exec -T ollama ollama create hrp-qa -f /Modelfile
fi
mkdir -p results
curl --fail --silent --show-error --max-time 180 http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"hrp-qa","messages":[{"role":"user","content":"什么是HRP系统？"}],"stream":false,"options":{"temperature":0,"num_predict":256,"num_ctx":1024}}' \
  -o results/ollama_smoke.json
python3 -c 'import json; from pathlib import Path; r=json.loads(Path("results/ollama_smoke.json").read_text()); assert r.get("done") and r.get("message", {}).get("content", "").strip(), "Ollama 未返回完整回答"; print(r["message"]["content"])'
echo "已导入并收到真实回答，记录保存到 results/ollama_smoke.json。"
echo "本机对话页：HRP_LOCAL_OLLAMA=1 streamlit run streamlit_app.py"
