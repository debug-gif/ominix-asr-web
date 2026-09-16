#!/bin/bash
# Download the two models required by ominix-asr-web.
# Uses hf-mirror.com by default; pass HUGGINGFACE_CO as the base URL if you prefer.
set -e

BASE="${HF_BASE:-https://hf-mirror.com}"
MODELS="$HOME/.OminiX/models"

download() {
    local repo="$1" dir="$2" file="$3"
    mkdir -p "$MODELS/$dir"
    echo "== $repo/$file -> $MODELS/$dir/$file"
    curl -sL -f -o "$MODELS/$dir/$file" "$BASE/$repo/resolve/main/$file"
}

echo "Downloading ASR model (Qwen3-ASR-0.6B-8bit, ~1.0 GB)..."
for f in config.json vocab.json merges.txt tokenizer_config.json model.safetensors; do
    download "mlx-community/Qwen3-ASR-0.6B-8bit" "qwen3-asr-0.6b" "$f"
done

echo "Downloading polish model (Qwen3-1.7B-4bit, ~0.9 GB)..."
for f in config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors model.safetensors.index.json; do
    download "mlx-community/Qwen3-1.7B-4bit" "qwen3-1.7b-4bit" "$f"
done

echo "Downloading summary model (Qwen3-4B-4bit, ~2.3 GB)..."
for f in config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors model.safetensors.index.json; do
    download "mlx-community/Qwen3-4B-4bit" "qwen3-4b-4bit" "$f"
done

echo "Done. Models installed under $MODELS"
