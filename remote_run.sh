#!/bin/bash
# remote_run.sh - Run Python scripts on remote machines with log streaming
#
# Usage: ./remote_run.sh [-d] [-n SESSION] [-b BRANCH] [-t HF_TOKEN] <ssh-host> <script-path> [script-args...]
#        ./remote_run.sh <ssh-host> <command>
#
# Options:
#   -d          Detach mode (don't attach to tmux session)
#   -n SESSION  Set tmux session name (default: jax_fusion)
#   -b BRANCH   Checkout specific git branch
#   -t TOKEN    Set HuggingFace token (HF_TOKEN env var)
#
# Commands: attach, stream, status, stop

set -e

if [[ "$TERM_PROGRAM" == "ghostty" ]]; then
    export TERM=xterm-256color
fi

REMOTE_DIR="~/jax_fusion"
REPO_URL="https://github.com/novastar53/jax-flow.git"
JAXPT_REPO_URL="https://github.com/novastar53/jaxpt.git"
DEFAULT_SESSION="jax_fusion"
LOG_DIR="remote_logs"
REMOTE_LOG_DIR="~/.cache/jax_fusion"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DETACH=false
SESSION_NAME="$DEFAULT_SESSION"
GIT_REF=""
HF_TOKEN=""

while getopts "dn:b:t:" opt; do
    case $opt in
        d) DETACH=true ;;
        n) SESSION_NAME="$OPTARG" ;;
        b) GIT_REF="$OPTARG" ;;
        t) HF_TOKEN="$OPTARG" ;;
        \?) echo -e "${RED}Invalid option: -$OPTARG${NC}" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

SSH_HOST=${1:-}
SECOND_ARG=${2:-}

if [ -z "$SSH_HOST" ]; then
    echo -e "${RED}Error: SSH host is required${NC}"
    echo "Usage: $0 [-d] [-n SESSION] [-b BRANCH] <ssh-host> <script-path>"
    exit 1
fi

COMMANDS="attach|stream|status|stop"
if [[ "$SECOND_ARG" =~ ^($COMMANDS)$ ]]; then
    COMMAND="$SECOND_ARG"
    SCRIPT_PATH=""
    SCRIPT_ARGS=""
else
    COMMAND="run"
    SCRIPT_PATH="$SECOND_ARG"
    shift 2
    SCRIPT_ARGS="$@"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

remote_exec() { ssh "$SSH_HOST" "$@"; }

validate_script() {
    if [[ ! "$1" =~ \.py$ ]]; then
        echo -e "${RED}Error: Script must be a .py file${NC}"
        exit 1
    fi
}

setup_remote() {
    echo -e "${GREEN}Setting up remote environment...${NC}"
    remote_exec "bash -l" << EOF
set -e
REMOTE_DIR=\$(eval echo ~/jax_fusion)
JAXPT_DIR=\$(eval echo ~/jaxpt)

# Clone/update jaxpt dependency in parent directory
if [ -d "\$JAXPT_DIR" ]; then
    echo "Updating jaxpt..."
    cd "\$JAXPT_DIR" && git fetch origin && git checkout main && git pull origin main
else
    echo "Cloning jaxpt to parent directory..."
    git clone $JAXPT_REPO_URL "\$JAXPT_DIR"
fi

# Clone/update jax_fusion
if [ -d "\$REMOTE_DIR" ]; then
    cd "\$REMOTE_DIR" && git fetch origin && git checkout main && git pull origin main
else
    git clone $REPO_URL "\$REMOTE_DIR" && cd "\$REMOTE_DIR"
fi
[ -n "$GIT_REF" ] && git checkout "$GIT_REF"
mkdir -p $REMOTE_LOG_DIR
[ -f "\$HOME/.local/bin/env" ] && source "\$HOME/.local/bin/env"
if ! command -v tmux &> /dev/null && command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y tmux
fi
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "\$HOME/.local/bin/env" 2>/dev/null
uv sync --extra cuda
echo "Setup complete!"
EOF
}

run_background() {
    local log_file="$REMOTE_LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"
    echo -e "${GREEN}Starting script in background...${NC}"

    # Build HF_TOKEN export if provided
    local hf_export=""
    if [ -n "$HF_TOKEN" ]; then
        hf_export="export HF_TOKEN='$HF_TOKEN' && "
    fi

    remote_exec "bash -l" << EOF
set -e
REMOTE_DIR=\$(eval echo ~/jax_fusion)
cd "\$REMOTE_DIR"
source \$HOME/.local/bin/env 2>/dev/null
tmux kill-session -t $SESSION_NAME 2>/dev/null || true
tmux new-session -d -s $SESSION_NAME -c "\$REMOTE_DIR"
tmux send-keys -t $SESSION_NAME "source \$HOME/.local/bin/env 2>/dev/null" Enter
tmux send-keys -t $SESSION_NAME "${hf_export}PYTHONUNBUFFERED=1 uv run python $SCRIPT_PATH $SCRIPT_ARGS 2>&1 | tee $log_file" Enter
echo "Session '$SESSION_NAME' started"
EOF
    echo -e "${GREEN}Script running.${NC} Use: $0 $SSH_HOST attach"
}

cmd_attach() {
    ssh -t "$SSH_HOST" "tmux attach-session -t $SESSION_NAME"
}

cmd_stream() {
    local log=$(remote_exec "ls -t $REMOTE_LOG_DIR/${SESSION_NAME}_*.log 2>/dev/null | head -1")
    [ -z "$log" ] && { echo -e "${RED}No logs found${NC}"; exit 1; }
    ssh "$SSH_HOST" "tail -f $log"
}

cmd_status() {
    remote_exec "bash -l" << EOF
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then echo "Tmux: RUNNING"; else echo "Tmux: NOT RUNNING"; fi
pgrep -f "uv run python" > /dev/null && echo "Python: RUNNING" || echo "Python: NOT RUNNING"
ls -t $REMOTE_LOG_DIR/${SESSION_NAME}_*.log 2>/dev/null | head -3 || echo "No logs"
EOF
}

cmd_stop() {
    echo -e "${GREEN}Stopping script...${NC}"
    remote_exec "bash -l" << EOF
tmux send-keys -t $SESSION_NAME C-c 2>/dev/null || true
sleep 2
tmux kill-session -t $SESSION_NAME 2>/dev/null || true
EOF
    mkdir -p "$LOG_DIR"
    local log=$(remote_exec "ls -t $REMOTE_LOG_DIR/${SESSION_NAME}_*.log 2>/dev/null | head -1" || echo "")
    [ -n "$log" ] && scp "$SSH_HOST:$log" "$LOG_DIR/" 2>/dev/null && echo -e "${GREEN}Logs saved${NC}"
}

case "$COMMAND" in
    run) validate_script "$SCRIPT_PATH"; setup_remote; run_background ;;
    attach) cmd_attach ;;
    stream) cmd_stream ;;
    status) cmd_status ;;
    stop) cmd_stop ;;
    *) echo "Unknown command: $COMMAND"; exit 1 ;;
esac
