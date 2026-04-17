#!/usr/bin/env bash
# =============================================================================
# push_file.sh  —  Transfer a SINGLE FILE from local to remote cluster.
#
# Usage:
#   ./push_file.sh -f path/to/local/file.py
#   ./push_file.sh -f path/to/local/file.py -c /custom/.transfer_config
#
# Remote path mirrors local path relative to LOCAL_BASE_DIR / REMOTE_BASE_DIR.
# =============================================================================
set -euo pipefail

export PATH="$HOME/bin:$PATH"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONFIG_FILE=""
LOCAL_FILE=""

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 -f <local_file> [-c <config_file>]"
    exit 1
}

while getopts ":f:c:" opt; do
    case $opt in
        f) LOCAL_FILE="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        *) usage ;;
    esac
done

[ -z "$LOCAL_FILE" ] && usage

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/.transfer_config}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi
chmod 600 "$CONFIG_FILE"
# shellcheck source=/dev/null
source "$CONFIG_FILE"

# ---------------------------------------------------------------------------
# Validate config
# ---------------------------------------------------------------------------
: "${REMOTE_USER:?'REMOTE_USER not set in config'}"
: "${REMOTE_HOST:?'REMOTE_HOST not set in config'}"
: "${REMOTE_BASE_DIR:?'REMOTE_BASE_DIR not set in config'}"

LOCAL_BASE_DIR="${LOCAL_BASE_DIR:-$SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Validate source file
# ---------------------------------------------------------------------------
if [ ! -f "$LOCAL_FILE" ]; then
    echo "Error: local file not found: $LOCAL_FILE"
    exit 1
fi
LOCAL_FILE="$(realpath "$LOCAL_FILE")"

# ---------------------------------------------------------------------------
# Build relative path
# ---------------------------------------------------------------------------
REAL_BASE="$(realpath "$LOCAL_BASE_DIR")"
REL_PATH="${LOCAL_FILE#"$REAL_BASE/"}"
if [ "$REL_PATH" = "$LOCAL_FILE" ]; then
    echo "Warning: file is outside LOCAL_BASE_DIR — using filename only."
    REL_PATH="$(basename "$LOCAL_FILE")"
fi

REMOTE_TARGET_DIR="$REMOTE_BASE_DIR/$(dirname "$REL_PATH")"
REMOTE_TARGET_DIR="${REMOTE_TARGET_DIR%/.}"

# ---------------------------------------------------------------------------
# Build SSH/SCP command arrays
# ---------------------------------------------------------------------------
SSH_OPTS=(-o StrictHostKeyChecking=no)

if [ -n "${SSH_KEY_PATH:-}" ]; then
    SSH_OPTS+=(-i "$SSH_KEY_PATH" -o BatchMode=yes)
    SSH_BASE=(ssh "${SSH_OPTS[@]}")
    SCP_BASE=(scp -C "${SSH_OPTS[@]}")
elif [ -n "${REMOTE_PASSWORD:-}" ]; then
    command -v sshpass >/dev/null 2>&1 || { echo "Error: sshpass not found."; exit 1; }
    SSH_OPTS+=(-o BatchMode=no)
    SSH_BASE=(sshpass -p "$REMOTE_PASSWORD" ssh "${SSH_OPTS[@]}")
    SCP_BASE=(sshpass -p "$REMOTE_PASSWORD" scp -C "${SSH_OPTS[@]}")
else
    SSH_OPTS+=(-o BatchMode=no)
    SSH_BASE=(ssh "${SSH_OPTS[@]}")
    SCP_BASE=(scp -C "${SSH_OPTS[@]}")
fi

REMOTE="$REMOTE_USER@$REMOTE_HOST"

# ---------------------------------------------------------------------------
# Pack → transfer → unpack
# ---------------------------------------------------------------------------
TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

TAR_NAME="push_file_$$.tar"
TAR_PATH="$TMPDIR_LOCAL/$TAR_NAME"

echo "==> Packing: $LOCAL_FILE"
tar -cvf "$TAR_PATH" -C "$(dirname "$LOCAL_FILE")" "$(basename "$LOCAL_FILE")"

echo "==> Creating remote dir: $REMOTE:$REMOTE_TARGET_DIR"
"${SSH_BASE[@]}" "$REMOTE" "mkdir -p '$REMOTE_TARGET_DIR'"

echo "==> Uploading to $REMOTE:$REMOTE_TARGET_DIR/$TAR_NAME"
"${SCP_BASE[@]}" "$TAR_PATH" "$REMOTE:$REMOTE_TARGET_DIR/$TAR_NAME"

echo "==> Extracting on remote and removing tarball"
"${SSH_BASE[@]}" "$REMOTE" "tar -xvf '$REMOTE_TARGET_DIR/$TAR_NAME' -C '$REMOTE_TARGET_DIR' && rm -f '$REMOTE_TARGET_DIR/$TAR_NAME'"

echo "==> Done: $REMOTE:$REMOTE_TARGET_DIR/$(basename "$LOCAL_FILE")"
