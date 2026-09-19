#!/usr/bin/env bash
# Stand up a local Ornith server that this harness can talk to.
#
#   ./scripts/serve-ornith.sh vllm     # single GPU, best throughput
#   ./scripts/serve-ornith.sh sglang
#   ./scripts/serve-ornith.sh llamacpp # GGUF, CPU or Metal
#   ./scripts/serve-ornith.sh ollama
#
# The two parser flags matter more than they look. Ornith is a reasoning
# model, so without --reasoning-parser the <think> block arrives inside the
# answer and every arm's parsing looks broken for reasons that have nothing
# to do with the experiment. Without the tool-call parser the agentic suite
# never sees a tool call at all.

set -euo pipefail

RUNTIME="${1:-vllm}"
MODEL="${ORNITH_HF_MODEL:-ornith-ai/Ornith-1.5-9B}"
GGUF="${ORNITH_GGUF_MODEL:-ornith-ai/Ornith-1.5-9B-GGUF}"
PORT="${ORNITH_PORT:-8000}"
CTX="${ORNITH_CTX:-262144}"

case "$RUNTIME" in
  vllm)
    exec vllm serve "$MODEL" \
      --served-model-name "$(basename "$MODEL")" \
      --host 0.0.0.0 --port "$PORT" \
      --max-model-len "$CTX" \
      --gpu-memory-utilization 0.90 \
      --enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      --reasoning-parser qwen3 \
      --trust-remote-code
    ;;
  sglang)
    exec python -m sglang.launch_server \
      --model-path "$MODEL" \
      --served-model-name "$(basename "$MODEL")" \
      --host 0.0.0.0 --port "$PORT" \
      --context-length "$CTX" \
      --mem-fraction-static 0.85 \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3
    ;;
  llamacpp)
    # Constrained decoding works here via GBNF, but confirm it took: a
    # non-zero violation count in the `constrained` arm means the server
    # ignored the schema and that arm silently degraded into `solo`.
    exec llama-server -hf "$GGUF" --port "$PORT" -c "$CTX" --jinja
    ;;
  ollama)
    echo "Ollama does not take a schema the way the constrained arm needs."
    echo "Run the solo and jev arms against it, or use vLLM for all three:"
    echo "  job run --arms solo,jev"
    echo
    exec ollama run "hf.co/${GGUF}"
    ;;
  *)
    echo "unknown runtime: $RUNTIME (want vllm, sglang, llamacpp or ollama)" >&2
    exit 2
    ;;
esac
