#!/bin/bash
# His2Trans Engine Test Runner
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh --quick      # Unit tests only (no framework needed)
#   ./run_tests.sh --ohos       # OHOS project tests
#   ./run_tests.sh --full       # Full integration (requires framework + API key)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

# Activate conda environment
if [ -f "/data/home/zhangxj/anaconda3/envs/c2r_frame/bin/python" ]; then
    PYTHON="/data/home/zhangxj/anaconda3/envs/c2r_frame/bin/python"
elif command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
    conda activate c2r_frame 2>/dev/null
    PYTHON="python"
else
    PYTHON="python3"
fi

echo "=== His2Trans Engine Tests ==="
echo "Python: $PYTHON"
echo "Backend: $BACKEND_DIR"
echo ""

# Parse mode
MODE="${1:-all}"
case "$MODE" in
    --quick|-q)
        echo "[Quick Mode] Running unit tests only (no framework needed)..."
        $PYTHON -m pytest tests/test_env_mapper.py tests/test_runner.py -v
        ;;
    --ohos|-o)
        echo "[OHOS Mode] Running OHOS project tests..."
        $PYTHON -m pytest tests/test_his2trans_engine.py -v -k "ohos or env" -m "not slow"
        ;;
    --standard|-s)
        echo "[Standard C Mode] Running standard C project tests..."
        $PYTHON -m pytest tests/test_his2trans_engine.py -v -k "standard or simulated"
        ;;
    --framework|-f)
        echo "[Framework Mode] Testing framework availability..."
        $PYTHON -m pytest tests/test_his2trans_engine.py -v -k "framework"
        ;;
    --full)
        echo "[Full Mode] Running all tests including slow integration tests..."
        $PYTHON -m pytest tests/ -v --timeout=3600
        ;;
    all|*)
        echo "[Default] Running all non-slow tests..."
        $PYTHON -m pytest tests/ -v -m "not slow"
        ;;
esac

echo ""
echo "=== Tests Complete ==="
