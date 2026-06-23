#!/usr/bin/env bash
# =============================================================================
# push_repo.sh  —  Transfer the whole 02_rl_control repo to ~/thesis/02_rl_control
# on the remote server. Excludes .claude/, .venv/, data_train/, *.whl.
#
# Usage:
#   ./push_repo.sh
#   ./push_repo.sh -c /custom/.transfer_config
# =============================================================================
set -euo pipefail

export PATH="$HOME/bin:$PATH"

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
CONFIG_FILE=""
while getopts ":c:" opt; do
    case $opt in
        c) CONFIG_FILE="$OPTARG" ;;
        *) echo "Usage: $0 [-c <config_file>]"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"   # bash_scripts/ lives inside the repo root
REPO_NAME="$(basename "$REPO_DIR")"   # 02_rl_control

CONFIG_FILE="${CONFIG_FILE:-$REPO_DIR/.transfer_config}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi
chmod 600 "$CONFIG_FILE"
# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${REMOTE_USER:?'REMOTE_USER not set in config'}"
: "${REMOTE_HOST:?'REMOTE_HOST not set in config'}"

REMOTE="$REMOTE_USER@$REMOTE_HOST"
# Relative path; scp/ssh interpret it relative to remote home dir.
# scp does NOT expand $HOME or ~ in destination strings reliably.
REMOTE_DEST_PARENT="thesis"
REMOTE_DEST="~/$REMOTE_DEST_PARENT/$REPO_NAME"

# ---------------------------------------------------------------------------
# Build SSH/SCP command arrays (same logic as push_dir.sh)
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

# ---------------------------------------------------------------------------
# Pack with excludes
# ---------------------------------------------------------------------------
TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

TAR_NAME="push_repo_$$.tar"
TAR_PATH="$TMPDIR_LOCAL/$TAR_NAME"

echo "==> Packing repo: $REPO_DIR"
echo "    Excludes: .claude/, .venv/, data_train/, *.whl"
tar -cvf "$TAR_PATH" \
    --exclude="$REPO_NAME/.claude" \
    --exclude="$REPO_NAME/.venv" \
    --exclude="$REPO_NAME/data_train" \
    --exclude="*.whl" \
    -C "$(dirname "$REPO_DIR")" "$REPO_NAME"

# ---------------------------------------------------------------------------
# Transfer + extract
# ---------------------------------------------------------------------------
echo "==> Creating remote parent: $REMOTE:$REMOTE_DEST_PARENT"
"${SSH_BASE[@]}" "$REMOTE" "mkdir -p $REMOTE_DEST_PARENT"

echo "==> Uploading tarball to $REMOTE:$REMOTE_DEST_PARENT/$TAR_NAME"
"${SCP_BASE[@]}" "$TAR_PATH" "$REMOTE:$REMOTE_DEST_PARENT/$TAR_NAME"

echo "==> Extracting on remote and removing tarball"
"${SSH_BASE[@]}" "$REMOTE" "tar -xvf $REMOTE_DEST_PARENT/$TAR_NAME -C $REMOTE_DEST_PARENT && rm -f $REMOTE_DEST_PARENT/$TAR_NAME"

echo "==> Done: $REMOTE:$REMOTE_DEST"
