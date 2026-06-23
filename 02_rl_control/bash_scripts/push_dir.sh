#!/usr/bin/env bash
# =============================================================================
# push_dir.sh  —  Transfer a DIRECTORY (recursively) from local to remote.
#
# Usage:
#   ./push_dir.sh -d path/to/local/dir
#   ./push_dir.sh -d path/to/local/dir -c /custom/.transfer_config
#
# Remote path mirrors local path relative to LOCAL_BASE_DIR / REMOTE_BASE_DIR.
# Remote contents are replaced (tar extract overwrites existing files).
# =============================================================================
set -euo pipefail

export PATH="$HOME/bin:$PATH"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONFIG_FILE=""
LOCAL_DIR=""

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 -d <local_dir> [-c <config_file>]"
    exit 1
}

while getopts ":d:c:" opt; do
    case $opt in
        d) LOCAL_DIR="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        *) usage ;;
    esac
done

[ -z "$LOCAL_DIR" ] && usage

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"   # bash_scripts/ lives inside the repo root
CONFIG_FILE="${CONFIG_FILE:-$REPO_DIR/.transfer_config}"

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

LOCAL_BASE_DIR="${LOCAL_BASE_DIR:-$REPO_DIR}"

# ---------------------------------------------------------------------------
# Validate source dir
# ---------------------------------------------------------------------------
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: local directory not found: $LOCAL_DIR"
    exit 1
fi
LOCAL_DIR="$(realpath "$LOCAL_DIR")"

# ---------------------------------------------------------------------------
# Build relative path
# ---------------------------------------------------------------------------
REAL_BASE="$(realpath "$LOCAL_BASE_DIR")"
REL_PATH="${LOCAL_DIR#"$REAL_BASE/"}"
if [ "$REL_PATH" = "$LOCAL_DIR" ]; then
    echo "Warning: dir is outside LOCAL_BASE_DIR — using dirname only."
    REL_PATH="$(basename "$LOCAL_DIR")"
fi

REMOTE_PARENT="$REMOTE_BASE_DIR/$(dirname "$REL_PATH")"
REMOTE_PARENT="${REMOTE_PARENT%/.}"

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

DIR_NAME="$(basename "$LOCAL_DIR")"
TAR_NAME="push_dir_$$.tar"
TAR_PATH="$TMPDIR_LOCAL/$TAR_NAME"

echo "==> Packing: $LOCAL_DIR"
tar -cvf "$TAR_PATH" -C "$(dirname "$LOCAL_DIR")" "$DIR_NAME"

echo "==> Creating remote parent: $REMOTE:$REMOTE_PARENT"
"${SSH_BASE[@]}" "$REMOTE" "mkdir -p '$REMOTE_PARENT'"

echo "==> Uploading to $REMOTE:$REMOTE_PARENT/$TAR_NAME"
"${SCP_BASE[@]}" "$TAR_PATH" "$REMOTE:$REMOTE_PARENT/$TAR_NAME"

echo "==> Extracting on remote and removing tarball"
"${SSH_BASE[@]}" "$REMOTE" "tar -xvf '$REMOTE_PARENT/$TAR_NAME' -C '$REMOTE_PARENT' && rm -f '$REMOTE_PARENT/$TAR_NAME'"

echo "==> Done: $REMOTE:$REMOTE_PARENT/$DIR_NAME"
