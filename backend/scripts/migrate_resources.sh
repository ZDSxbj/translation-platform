#!/bin/bash
# Migrate resources from His2Trans-Opt- to translation-platform/backend/data/
# Uses symlinks to save disk space (~330 MB saved).
set -e

HIS2TRANS_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)/His2Trans-Opt-"
PLATFORM_DATA="$(cd "$(dirname "$0")/.." && pwd)/data"

echo "=== Migrating His2Trans Resources ==="
echo "Source: $HIS2TRANS_ROOT"
echo "Target: $PLATFORM_DATA"
echo ""

# -- RAG Knowledge Base --
echo "[1/4] RAG knowledge base..."
mkdir -p "$PLATFORM_DATA/rag"
ln -sf "$HIS2TRANS_ROOT/framework/workspace/rag/knowledge_base.json" "$PLATFORM_DATA/rag/knowledge_base.json"
ln -sf "$HIS2TRANS_ROOT/framework/workspace/rag/bm25_index.pkl" "$PLATFORM_DATA/rag/bm25_index.pkl"
echo "  ✓ knowledge_base.json (173 MB)"
echo "  ✓ bm25_index.pkl (11 MB)"

# -- OpenHarmony SDK --
echo "[2/4] OpenHarmony SDK..."
mkdir -p "$PLATFORM_DATA/ohos"
ln -sfT "$HIS2TRANS_ROOT/data/ohos/ohos_root_min" "$PLATFORM_DATA/ohos/ohos_root_min"
echo "  ✓ ohos_root_min (92 MB)"

# -- NLTK Data --
echo "[3/4] NLTK data..."
mkdir -p "$PLATFORM_DATA/nltk_data"
if [ -d "$HIS2TRANS_ROOT/framework/data/nltk_data" ]; then
    for subdir in "$HIS2TRANS_ROOT/framework/data/nltk_data"/*/; do
        name=$(basename "$subdir")
        ln -sfT "$subdir" "$PLATFORM_DATA/nltk_data/$name"
        echo "  ✓ nltk_data/$name"
    done
else
    echo "  ⚠ nltk_data not found, skipping"
fi

# -- Prompt Templates --
echo "[4/4] Prompt templates..."
mkdir -p "$PLATFORM_DATA/prompts"
for f in "$HIS2TRANS_ROOT/framework/generate"/*.txt; do
    if [ -f "$f" ]; then
        ln -sf "$f" "$PLATFORM_DATA/prompts/$(basename "$f")"
        echo "  ✓ $(basename "$f")"
    fi
done

echo ""
echo "=== Migration complete ==="
echo "All resources symlinked to $PLATFORM_DATA"
