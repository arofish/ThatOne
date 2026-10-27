
#!/usr/bin/env bash
set -Eeuo pipefail

# ---------- 定位项目与 Python ----------

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-app/bin/python"

TOKEN_DIR="$HOME/.config/thatone"
TOKEN_FILE="$TOKEN_DIR/dev-token"

if [[ ! -x "$PY" ]]; then
    printf 'Missing Python: %s\n' "$PY" >&2
    exit 1
fi

# ---------- 先检查端口，再生成 Token ----------

if lsof -nP -iTCP:8765 -sTCP:LISTEN -t | grep -q .; then
    printf 'Port 8765 is already in use.\n' >&2
    printf 'Stop the existing service first.\n' >&2
    exit 1
fi

# ---------- 准备私有配置目录 ----------

umask 077

mkdir -p "$TOKEN_DIR"
chmod 700 "$TOKEN_DIR"

server_pid=""
tmp_token=""

# ---------- 进程退出时清理 ----------

cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi

    # 只清理属于本次启动的 Token。
    if [[ -f "$TOKEN_FILE" ]] && \
       [[ "$(cat "$TOKEN_FILE")" == "${THATONE_API_TOKEN:-}" ]]; then
        rm -f "$TOKEN_FILE"
    fi

    if [[ -n "$tmp_token" ]]; then
        rm -f "$tmp_token"
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------- 生成本次运行的随机 Token ----------

tmp_token="$(mktemp "$TOKEN_DIR/.dev-token.XXXXXX")"

"$PY" -c \
  'import secrets; print(secrets.token_urlsafe(32))' \
  > "$tmp_token"

chmod 600 "$tmp_token"

export THATONE_API_TOKEN="$(cat "$tmp_token")"

mv -f "$tmp_token" "$TOKEN_FILE"
tmp_token=""

# ---------- 固定离线运行环境 ----------

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

printf 'Starting ThatOne local backend...\n'
printf 'Address: http://127.0.0.1:8765\n'
printf 'Token file: %s (content hidden)\n' "$TOKEN_FILE"

# ---------- 启动并保持后端运行 ----------

"$PY" -m photoindex.api &
server_pid=$!

wait "$server_pid"
