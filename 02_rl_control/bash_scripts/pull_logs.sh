#!/usr/bin/env bash
# =============================================================================
# pull_logs.sh  —  Pull the logs/ directory from remote to local.
#
# Usage:
#   ./pull_logs.sh                        # pulls to SCRIPT_DIR/logs/
#   ./pull_logs.sh -o /custom/local/dest  # custom local destination
#   ./pull_logs.sh -r logs                # custom remote subpath under REMOTE_BASE_DIR
#   ./pull_logs.sh -c /custom/.transfer_config
#
# Merges into local destination (tar extract, does not wipe local logs first).
# =============================================================================
set -euo pipefail

# Ensure ~/bin is in PATH (for locally-installed sshpass)
export PATH="$HOME/bin:$PATH"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONFIG_FILE=""
REMOTE_LOGS_SUBPATH="logs"   # relative to REMOTE_BASE_DIR
LOCAL_DEST=""

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 [-r <remote_logs_subpath>] [-o <local_dest>] [-c <config_file>]"
    echo "  -r   Subpath under REMOTE_BASE_DIR to pull (default: logs)"
    echo "  -o   Local destination directory (default: SCRIPT_DIR/logs_cluster)"
    echo "  -c   Config file path"
    exit 1
}

while getopts ":r:o:c:" opt; do
    case $opt in
        r) REMOTE_LOGS_SUBPATH="$OPTARG" ;;
        o) LOCAL_DEST="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        *) usage ;;
    esac
done

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
# Resolve paths
# ---------------------------------------------------------------------------
REMOTE_LOGS_DIR="$REMOTE_BASE_DIR/$REMOTE_LOGS_SUBPATH"
REMOTE_LOGS_PARENT="$(dirname "$REMOTE_LOGS_DIR")"
REMOTE_LOGS_DIRNAME="$(basename "$REMOTE_LOGS_DIR")"

LOCAL_DEST="${LOCAL_DEST:-$LOCAL_BASE_DIR/logs_cluster}"
mkdir -p "$LOCAL_DEST"

# ---------------------------------------------------------------------------
# Build SSH/SCP command arrays (avoids all quoting/eval issues)
# ---------------------------------------------------------------------------
SSH_OPTS=(-o StrictHostKeyChecking=no)

if [ -n "${SSH_KEY_PATH:-}" ]; then
    SSH_OPTS+=(-i "$SSH_KEY_PATH" -o BatchMode=yes)
    SSH_BASE=(ssh "${SSH_OPTS[@]}")
    SCP_BASE=(scp -C "${SSH_OPTS[@]}")
elif [ -n "${REMOTE_PASSWORD:-}" ]; then
    command -v sshpass >/dev/null 2>&1 || {
        echo "Error: sshpass not found. Install it or place it in ~/bin/"
        exit 1
    }
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
# Pack on remote → transfer → unpack locally
# ---------------------------------------------------------------------------
TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

TAR_NAME="pull_logs_$$.tar"
REMOTE_TMP_TAR="/tmp/$TAR_NAME"

echo "==> Packing on remote: $REMOTE:$REMOTE_LOGS_DIR"
"${SSH_BASE[@]}" "$REMOTE" "tar -cvf '$REMOTE_TMP_TAR' -C '$REMOTE_LOGS_PARENT' '$REMOTE_LOGS_DIRNAME'"

echo "==> Transferring to local: $TMPDIR_LOCAL/$TAR_NAME"
"${SCP_BASE[@]}" "$REMOTE:$REMOTE_TMP_TAR" "$TMPDIR_LOCAL/$TAR_NAME"

echo "==> Cleaning up remote tarball"
"${SSH_BASE[@]}" "$REMOTE" "rm -f '$REMOTE_TMP_TAR'"

echo "==> Wiping previous local logs at: $LOCAL_DEST/$REMOTE_LOGS_DIRNAME"
rm -rf "$LOCAL_DEST/$REMOTE_LOGS_DIRNAME"

echo "==> Extracting to: $LOCAL_DEST"
tar -xvf "$TMPDIR_LOCAL/$TAR_NAME" -C "$LOCAL_DEST"

echo ""
echo "==> Done. Logs at: $LOCAL_DEST/$REMOTE_LOGS_DIRNAME"
echo ""
echo "==> TensorBoard:"
echo "    tensorboard --logdir $LOCAL_DEST/$REMOTE_LOGS_DIRNAME"
