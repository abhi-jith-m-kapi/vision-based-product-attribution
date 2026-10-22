#!/usr/bin/env bash
# Launch the Streamlit app. Override STREAMLIT_PORT or PYTHON if needed.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PORT="${STREAMLIT_PORT:-8501}"

if [[ ! -x "$PYTHON" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    else
        echo "error: Python was not found. Install Python or set PYTHON." >&2
        exit 1
    fi
fi

exec "$PYTHON" -m streamlit run "$ROOT/streamlit_app.py" --server.port "$PORT" "$@"
