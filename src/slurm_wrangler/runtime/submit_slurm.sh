#!/usr/bin/env bash
# slurm_wrangler: runs on the cluster LOGIN node. Reads SW_* environment
# variables (passed by `sw run` over ssh), generates a job script, submits it.
#
# The job body below is a port of UWLab's submit_job_slurm_hyak.sh machinery:
#   - USR1/TERM traps -> requeue on time limit / preemption
#   - bad-node registry -> --exclude at submit + bounce guard at attempt start
#   - crash classifier -> auto-requeue transient GPU/hardware faults only
#
# All SW_* configuration flows through `sbatch --export=ALL`, so the body needs
# no placeholder substitution — only the #SBATCH header is printed with values.
set -euo pipefail

req() { local v="${!1:-}"; [ -n "$v" ] || { echo "[sw-submit] ERROR: $1 is empty" >&2; exit 1; }; }
for v in SW_PROJECT SW_PROJECT_DIR SW_CODE_DIR SW_SIF SW_RUNTIME_DIR SW_JOB_NAME \
         SW_NODES SW_GPUS_PER_NODE SW_TIME SW_CPUS_PER_TASK \
         SW_MEM_DIRECTIVE SW_SLURM_LOGS_DIR SW_SIGNAL_SECS SW_CMD_B64 SW_WORKDIR; do
    req "$v"
done

# SLURM writes slurm-<id>.out into the submit dir if the log path is bad; fail loudly instead.
if [[ "$SW_SLURM_LOGS_DIR" != /* ]]; then
    echo "[sw-submit] ERROR: SW_SLURM_LOGS_DIR must be absolute, got '$SW_SLURM_LOGS_DIR'" >&2
    exit 1
fi
mkdir -p "$SW_SLURM_LOGS_DIR"

# Partition race: a sibling already started and claimed the group -> don't queue a late copy.
if [ -n "${SW_RACE_GROUP:-}" ] && [ -s "$SW_SLURM_LOGS_DIR/.race_${SW_RACE_GROUP}/owner" ]; then
    echo "[sw-submit] race group '$SW_RACE_GROUP' already claimed by job $(cat "$SW_SLURM_LOGS_DIR/.race_${SW_RACE_GROUP}/owner"); skipping submit."
    exit 0
fi

# --- Bad-node registry -> exclude list -----------------------------------------------
BAD_NODES_FILE="$SW_SLURM_LOGS_DIR/bad_nodes.tsv"
BAD_NODE_TTL_DAYS="${SW_BAD_NODE_TTL_DAYS:-7}"
BAD_NODE_MAX="${SW_BAD_NODE_MAX:-30}"
exclude_list="${SW_EXCLUDE_NODES:-}"
if [[ "${SW_SKIP_BAD_NODES:-0}" != "1" && -s "$BAD_NODES_FILE" ]]; then
    auto_excludes=$(awk -F'\t' -v now="$(date +%s)" -v ttl="$((BAD_NODE_TTL_DAYS * 86400))" '
        /^#/ || NF < 2 { next }
        (now - $2) < ttl { if ($2 > seen[$1]) seen[$1] = $2 }
        END { for (n in seen) printf "%s\t%s\n", seen[n], n }
    ' "$BAD_NODES_FILE" | sort -rn | head -n "$BAD_NODE_MAX" | cut -f2 | paste -sd, -)
    if [ -n "$auto_excludes" ]; then
        exclude_list="${exclude_list:+$exclude_list,}$auto_excludes"
    fi
fi

# ckpt-partition sanity check (Hyak names checkpoint accounts ckpt-<lab>)
if [[ "${SW_PARTITION:-}" == *"ckpt"* && -n "${SW_ACCOUNT:-}" ]]; then
    if [[ "$SW_ACCOUNT" != ckpt-* && "$SW_ACCOUNT" != *-ckpt ]]; then
        echo "[sw-submit] WARNING: partition '$SW_PARTITION' with account '$SW_ACCOUNT' (no ckpt marker)."
        echo "[sw-submit]          ckpt jobs usually need e.g. --account ckpt-weirdlab --qos ckpt-gpu."
    fi
fi

echo "----------------------------------------------------------------"
echo "slurm_wrangler submit: $SW_JOB_NAME"
echo "  project:    $SW_PROJECT"
echo "  code:       $SW_CODE_DIR"
echo "  image:      $SW_SIF"
echo "  nodes/gpus: ${SW_NODES}x${SW_GPUS_PER_NODE}  partition=${SW_PARTITION:-<default>} account=${SW_ACCOUNT:-<default>} qos=${SW_QOS:-<default>}"
echo "  time:       $SW_TIME    logs: $SW_SLURM_LOGS_DIR"
[ -n "$exclude_list" ] && echo "  excluded:   $exclude_list" || true
echo "----------------------------------------------------------------"

JOB_SH="$(mktemp "${TMPDIR:-/tmp}/sw_job_XXXXXX.sh")"
trap 'rm -f "$JOB_SH"' EXIT

# ---- #SBATCH header (real values) ---------------------------------------------------
{
    echo '#!/bin/bash'
    echo "#SBATCH --job-name=\"$SW_JOB_NAME\""
    echo "#SBATCH --output=$SW_SLURM_LOGS_DIR/%x-%j.out"
    echo "#SBATCH --error=$SW_SLURM_LOGS_DIR/%x-%j.err"
    echo "#SBATCH --open-mode=append"
    [ -n "${SW_ACCOUNT:-}" ]    && echo "#SBATCH --account=$SW_ACCOUNT" || true
    [ -n "${SW_PARTITION:-}" ] && echo "#SBATCH --partition=$SW_PARTITION" || true
    echo "#SBATCH --nodes=$SW_NODES"
    echo "#SBATCH --ntasks-per-node=1"
    if [ "${SW_GPUS_PER_NODE}" -gt 0 ]; then
        echo "#SBATCH --gpus-per-node=$SW_GPUS_PER_NODE"
    fi
    echo "#SBATCH --cpus-per-task=$SW_CPUS_PER_TASK"
    echo "$SW_MEM_DIRECTIVE"
    echo "#SBATCH --time=$SW_TIME"
    [ -n "${SW_QOS:-}" ]        && echo "#SBATCH --qos=$SW_QOS" || true
    [ -n "${SW_CONSTRAINT:-}" ] && echo "#SBATCH --constraint=\"$SW_CONSTRAINT\"" || true
    if [ "${SW_REQUEUE:-1}" = "1" ]; then
        echo "#SBATCH --requeue"
        echo "#SBATCH --signal=B:USR1@$SW_SIGNAL_SECS"
    fi
    [ -n "$exclude_list" ]      && echo "#SBATCH --exclude=$exclude_list" || true
} > "$JOB_SH"

# ---- job body (literal; reads SW_* via --export=ALL at runtime) ---------------------
cat >> "$JOB_SH" <<'JOB_BODY'

BAD_NODES_FILE="$SW_SLURM_LOGS_DIR/bad_nodes.tsv"
BAD_NODE_TTL_SECS=$(( ${SW_BAD_NODE_TTL_DAYS:-7} * 86400 ))

# --- Partition race: first copy to start claims the group, cancels pending siblings ---
if [ -n "${SW_RACE_GROUP:-}" ]; then
    CLAIM="$SW_SLURM_LOGS_DIR/.race_${SW_RACE_GROUP}"
    if mkdir "$CLAIM" 2>/dev/null; then
        echo "$SLURM_JOB_ID" > "$CLAIM/owner"
    elif [ "$(cat "$CLAIM/owner" 2>/dev/null)" != "$SLURM_JOB_ID" ]; then
        echo "[$(date)] Lost partition race (owner $(cat "$CLAIM/owner" 2>/dev/null)); exiting."
        exit 0
    fi
    echo "[$(date)] Won partition race; cancelling pending siblings of $SLURM_JOB_NAME"
    scancel --user="$USER" --name="$SLURM_JOB_NAME" --state=PENDING || true
fi

# --- Requeue handler: time limit (USR1) and preemption (TERM) ---
requeue_handler() {
    echo "[$(date)] Caught signal: $1 - marking job $SLURM_JOB_ID for requeue"
    scontrol requeue "$SLURM_JOB_ID"
    # Don't exit - let the job keep saving checkpoints until forced kill
}
if [ "${SW_REQUEUE:-1}" = "1" ]; then
    trap 'requeue_handler USR1' USR1
    trap 'requeue_handler TERM' TERM
    echo "[$(date)] Job $SLURM_JOB_ID starting on $(hostname); requeue handler installed"
else
    echo "[$(date)] Job $SLURM_JOB_ID starting on $(hostname); requeue disabled"
fi

# --- Bad-node bounce guard ---
# --exclude is fixed at submit time and SLURM refuses ExcNodeList updates on a
# RUNNING job, so each attempt re-checks its own allocation against the registry
# and requeues immediately off a known-bad node (seconds vs minutes of staging).
MAX_BAD_NODE_BOUNCES=${SW_MAX_BAD_NODE_BOUNCES:-10}
if [ "${SW_REQUEUE:-1}" = "1" ] && [ -s "$BAD_NODES_FILE" ]; then
    BOUNCE_FILE="$SW_SLURM_LOGS_DIR/.bad_node_bounce_${SLURM_JOB_ID}"
    BAD_HITS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | awk -F'\t' \
        -v now="$(date +%s)" -v ttl="$BAD_NODE_TTL_SECS" '
        FNR == NR { if ($0 !~ /^#/ && NF >= 2 && (now - $2) < ttl) bad[$1] = 1; next }
        ($1 in bad) { print $1 }
    ' "$BAD_NODES_FILE" - | sort -u | paste -sd, -)
    if [ -n "$BAD_HITS" ]; then
        BOUNCES=$(cat "$BOUNCE_FILE" 2>/dev/null || echo 0)
        if [ "$BOUNCES" -lt "$MAX_BAD_NODE_BOUNCES" ]; then
            echo $((BOUNCES + 1)) > "$BOUNCE_FILE"
            echo "[$(date)] Allocated known-bad node(s): $BAD_HITS. Bounce $((BOUNCES + 1))/$MAX_BAD_NODE_BOUNCES - requeuing."
            scontrol requeue "$SLURM_JOB_ID"
            sleep 10
            exit 0
        fi
        echo "[$(date)] Bad node(s) $BAD_HITS but bounce budget exhausted - running here anyway."
    fi
fi

# --- Distributed rendezvous ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(($SLURM_JOB_ID % 55535 + 10000))
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Record log sizes so the classifier scans only THIS attempt's output
# (--open-mode=append accumulates all requeue attempts in one file).
JOB_OUT="$SW_SLURM_LOGS_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
JOB_ERR="$SW_SLURM_LOGS_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
OUT_OFFSET=$(stat -c%s "$JOB_OUT" 2>/dev/null || echo 0)
ERR_OFFSET=$(stat -c%s "$JOB_ERR" 2>/dev/null || echo 0)

# --- Execute ---
srun --export=ALL bash "$SW_RUNTIME_DIR/run_container.sh" &
wait $!
EXIT_CODE=$?
echo "[$(date)] Job finished with exit code: $EXIT_CODE"

# --- Crash auto-requeue: transient hardware faults only ---
# SLURM does not auto-requeue application failures. Distinguish transient
# GPU/driver faults (retry on a fresh node) from deterministic code/config bugs
# (fail fast) by scanning this attempt's log slice.
if [ "$EXIT_CODE" -ne 0 ] && [ "${SW_REQUEUE:-1}" = "1" ]; then
    sleep 2  # let SLURM flush the log tail
    ATTEMPT_OUTPUT="$( { tail -c "+$((OUT_OFFSET + 1))" "$JOB_OUT" 2>/dev/null; \
                         tail -c "+$((ERR_OFFSET + 1))" "$JOB_ERR" 2>/dev/null; } )"
    HW_FAULT_RE="${SW_HW_FAULT_RE:-cudaErrorECCUncorrectable|uncorrectable ECC error|CUDA error 214|ERROR_INITIALIZATION_FAILED|No device could be created|Failed to enable P2P access|cudaErrorDevicesUnavailable|are busy or unavailable|Xid [0-9]|NVLink}"
    # Deterministic signatures take PRECEDENCE: a real bug won't be fixed by a
    # fresh node, even when an ECC signature co-occurs. Deliberately narrow - no
    # generic "Traceback", which hardware crashes also print.
    DETERMINISTIC_RE="${SW_DETERMINISTIC_RE:-ModuleNotFoundError|ImportError|OutOfMemoryError|CUDA out of memory|SyntaxError|IndentationError|NameError|AttributeError|TypeError|KeyError:|FileNotFoundError|ConfigCompositionException|omegaconf\.errors|hydra\.errors|argument .* invalid|unrecognized arguments}"
    if echo "$ATTEMPT_OUTPUT" | grep -qiE "$DETERMINISTIC_RE"; then
        echo "[$(date)] Non-zero exit ($EXIT_CODE): deterministic (code/config) failure signature - NOT requeuing."
    elif scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -qE '(^|[[:space:]])Requeue=0([[:space:]]|$)'; then
        echo "[$(date)] Non-zero exit ($EXIT_CODE) but Requeue=0 (operator cancel) - not auto-requeuing."
    elif echo "$ATTEMPT_OUTPUT" | grep -qiE "$HW_FAULT_RE"; then
        MAX_HW_REQUEUES=${SW_MAX_HW_REQUEUES:-5}
        HW_COUNTER_FILE="$SW_SLURM_LOGS_DIR/.hw_requeue_${SLURM_JOB_ID}"
        HW_COUNT=$(cat "$HW_COUNTER_FILE" 2>/dev/null || echo 0)
        # Record faulting node(s): prefer torchrun's "Root Cause" host, else the
        # whole allocation. Future submissions exclude them; the bounce guard
        # keeps requeued attempts off them.
        FAULT_NODES=$(echo "$ATTEMPT_OUTPUT" | grep -oE '^[[:space:]]*host[[:space:]]*:[[:space:]]*[A-Za-z0-9._-]+' \
            | sed -E 's/.*:[[:space:]]*//; s/\..*//' | sort -u)
        [ -z "$FAULT_NODES" ] && FAULT_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null)
        for bad_node in $FAULT_NODES; do
            printf '%s\t%s\t%s\t%s\n' "$bad_node" "$(date +%s)" "$SLURM_JOB_ID" "hw-fault" >> "$BAD_NODES_FILE"
            echo "[$(date)] Recorded bad node '$bad_node' in $BAD_NODES_FILE."
        done
        if [ "$HW_COUNT" -lt "$MAX_HW_REQUEUES" ]; then
            echo $((HW_COUNT + 1)) > "$HW_COUNTER_FILE"
            echo "[$(date)] Transient hardware fault (exit $EXIT_CODE). Requeue $((HW_COUNT + 1))/$MAX_HW_REQUEUES."
            scontrol requeue "$SLURM_JOB_ID"
            sleep 5
            exit 0
        fi
        echo "[$(date)] Hardware-fault requeue budget exhausted - giving up."
        rm -f "$HW_COUNTER_FILE"
    else
        echo "[$(date)] Non-zero exit ($EXIT_CODE), no hardware-fault signature - NOT requeuing."
    fi
fi

exit $EXIT_CODE
JOB_BODY

if [ -n "${SW_RACE_GROUP:-}" ] && [ -s "$SW_SLURM_LOGS_DIR/.race_${SW_RACE_GROUP}/owner" ]; then
    echo "[sw-submit] Race group '$SW_RACE_GROUP' already claimed by job $(cat "$SW_SLURM_LOGS_DIR/.race_${SW_RACE_GROUP}/owner"); not submitting partition=${SW_PARTITION:-<default>}."
    exit 0
fi

SUBMIT_OUT=$(sbatch --export=ALL "$JOB_SH")
echo "$SUBMIT_OUT"
