#!/usr/bin/env bash
# slurm_wrangler: runs on the COMPUTE node (one instance per node via srun).
# Stages the SIF + code snapshot + declared caches onto node-local scratch,
# assembles bind mounts, and launches the user's command inside the container.
# Generalized port of UWLab's run_singularity.sh.
#
# Everything arrives via SW_* environment variables (sbatch --export=ALL).
echo "(run_container.sh): starting on $(hostname)"

req() { local v="${!1:-}"; [ -n "$v" ] || { echo "[sw-run] ERROR: $1 is empty on compute node" >&2; exit 1; }; }
for v in SW_PROJECT SW_PROJECT_DIR SW_CODE_DIR SW_SIF SW_WORKDIR SW_CMD_B64 SW_LAUNCHER; do
    req "$v"
done

# Job-specific node-local dir: multiple jobs on one node must not share paths.
TMPDIR=${TMPDIR:-/tmp}
JOB_TMPDIR="$TMPDIR/sw_job_${SLURM_JOB_ID:-$$}"
rm -rf "$JOB_TMPDIR"
mkdir -p "$JOB_TMPDIR"
echo "[sw-run] node-local dir: $JOB_TMPDIR"

cleanup() { rm -rf "$JOB_TMPDIR"; }

# --- Stage the SIF (retry: GPFS reads can fail transiently) --------------------------
SIF_LOCAL="$JOB_TMPDIR/$(basename "$SW_SIF")"
MAX_RETRIES=3
for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[sw-run] staging image (attempt $attempt/$MAX_RETRIES): $SW_SIF"
    if cp "$SW_SIF" "$SIF_LOCAL" && [ -s "$SIF_LOCAL" ]; then
        break
    fi
    rm -f "$SIF_LOCAL"
    if [ "$attempt" -eq "$MAX_RETRIES" ]; then
        echo "[sw-run] ERROR: failed to stage image after $MAX_RETRIES attempts" >&2
        cleanup; exit 1
    fi
    sleep 5
done

# --- Stage the code snapshot (RW copy; the GPFS snapshot stays immutable) ------------
CODE_LOCAL="$JOB_TMPDIR/code"
cp -r "$SW_CODE_DIR" "$CODE_LOCAL" || { echo "[sw-run] ERROR: code staging failed" >&2; cleanup; exit 1; }

BIND_FLAGS=()
ENV_FLAGS=()

# --- Caches --------------------------------------------------------------------------
# SW_CACHES_B64: base64 of newline records  host_slug|container|writeback|mode|dir1,dir2
# Persistent copy: $SW_PROJECT_DIR/cache/<slug> ; staged once per slug to node-local.
declare -A STAGED_SLUG
CACHE_RECORDS=""
if [ -n "${SW_CACHES_B64:-}" ]; then
    CACHE_RECORDS="$(echo "$SW_CACHES_B64" | base64 -d)"
    while IFS='|' read -r slug container writeback mode ensure_dirs; do
        [ -z "$slug" ] && continue
        HOST_DIR="$SW_PROJECT_DIR/cache/$slug"
        LOCAL_DIR="$JOB_TMPDIR/cache/${slug//\//_}"
        if [ -z "${STAGED_SLUG[$slug]:-}" ]; then
            mkdir -p "$HOST_DIR"
            if [ -n "$ensure_dirs" ]; then
                IFS=',' read -ra EDS <<< "$ensure_dirs"
                for ed in "${EDS[@]}"; do mkdir -p "$HOST_DIR/$ed"; done
            fi
            mkdir -p "$LOCAL_DIR"
            rsync -a "$HOST_DIR/" "$LOCAL_DIR/" || echo "[sw-run] WARNING: cache stage failed for $slug (continuing empty)"
            if [ -n "$ensure_dirs" ]; then
                IFS=',' read -ra EDS <<< "$ensure_dirs"
                for ed in "${EDS[@]}"; do mkdir -p "$LOCAL_DIR/$ed"; done
            fi
            STAGED_SLUG[$slug]=1
        fi
        BIND_FLAGS+=(-B "$LOCAL_DIR:$container:rw")
    done <<< "$CACHE_RECORDS"
fi

# --- Static binds (persist dirs, data, cluster mount, ssl) ---------------------------
# SW_BINDS_B64: base64 of newline records  host::container::mode
if [ -n "${SW_BINDS_B64:-}" ]; then
    while IFS= read -r rec; do
        [ -z "$rec" ] && continue
        host="${rec%%::*}"; rest="${rec#*::}"
        container="${rest%%::*}"; mode="${rest##*::}"
        case "$host" in
            "$SW_PROJECT_DIR"/*) mkdir -p "$host" ;;
        esac
        if [ ! -e "$host" ]; then
            echo "[sw-run] WARNING: bind source missing, skipping: $host"
            continue
        fi
        BIND_FLAGS+=(-B "$host:$container:$mode")
    done < <(echo "$SW_BINDS_B64" | base64 -d; echo)
fi

# Code over the workdir
BIND_FLAGS+=(-B "$CODE_LOCAL:$SW_WORKDIR:rw")

# Persistent /tmp on the cluster FS (the --containall default /tmp is tiny tmpfs)
mkdir -p "$SW_PROJECT_DIR/tmp"
BIND_FLAGS+=(-B "$SW_PROJECT_DIR/tmp:/tmp:rw")

# DNS + CA certs (needed for wandb etc. under --containall on Rocky hosts)
[ -e /etc/resolv.conf ] && BIND_FLAGS+=(-B /etc/resolv.conf:/etc/resolv.conf:ro)
[ -d /etc/pki ] && BIND_FLAGS+=(-B /etc/pki:/etc/pki:ro)
[ -d /etc/ssl ] && BIND_FLAGS+=(-B /etc/ssl:/etc/ssl:ro)

# --- Forwarded environment -----------------------------------------------------------
# SW_ENV_FWD_B64: base64 of newline KEY=VALUE records
if [ -n "${SW_ENV_FWD_B64:-}" ]; then
    while IFS= read -r kv; do
        [ -z "$kv" ] && continue
        ENV_FLAGS+=(--env "$kv")
    done < <(echo "$SW_ENV_FWD_B64" | base64 -d; echo)
fi
ENV_FLAGS+=(--env "SW_JOB_ID=${SLURM_JOB_ID:-0}")
ENV_FLAGS+=(--env "SLURM_JOB_ID=${SLURM_JOB_ID:-0}")
if [ -d /etc/pki/tls/certs ]; then
    ENV_FLAGS+=(--env SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
                --env REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt \
                --env CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt)
fi

# --- Prestage hook (project script run on the node BEFORE the container starts) ------
if [ -n "${SW_PRESTAGE:-}" ]; then
    if [ -f "$CODE_LOCAL/$SW_PRESTAGE" ]; then
        echo "[sw-run] running prestage hook: $SW_PRESTAGE"
        (cd "$CODE_LOCAL" && SW_JOB_TMPDIR="$JOB_TMPDIR" bash "$SW_PRESTAGE") \
            || { echo "[sw-run] ERROR: prestage hook failed" >&2; cleanup; exit 1; }
    else
        echo "[sw-run] WARNING: prestage hook not found in code snapshot: $SW_PRESTAGE"
    fi
fi

# --- Assemble the inner command ------------------------------------------------------
USER_CMD="$(echo "$SW_CMD_B64" | base64 -d)"
GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-0}"

if [ "$SW_LAUNCHER" = "torchrun" ]; then
    # Convention: the command is "<script.py> [args]"; a leading python/python3 is stripped.
    STRIPPED_CMD="$USER_CMD"
    STRIPPED_CMD="${STRIPPED_CMD#python3 }"
    STRIPPED_CMD="${STRIPPED_CMD#python }"
    INNER="cd $SW_WORKDIR && exec python -m torch.distributed.run \
--nnodes=${SLURM_NNODES:-1} --nproc_per_node=${GPUS_ON_NODE:-1} \
--rdzv_backend=c10d --rdzv_id=${SLURM_JOB_ID:-0} --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
$STRIPPED_CMD"
else
    INNER="cd $SW_WORKDIR && $USER_CMD"
fi

if [ "${SW_IMAGE_KIND:-conda}" = "conda" ]; then
    # micromamba run applies full env activation (incl. activate.d scripts:
    # CUDA_HOME, LD_LIBRARY_PATH, ...), not just PATH.
    RUN_INSIDE=(micromamba run -p /opt/env bash -c "$INNER")
else
    RUN_INSIDE=(bash -c "$INNER")
fi

# --- Launch --------------------------------------------------------------------------
SING_BIN=$(command -v singularity || command -v apptainer) \
    || { echo "[sw-run] ERROR: neither singularity nor apptainer on PATH" >&2; cleanup; exit 1; }
NV_FLAG=()
[ "$GPUS_ON_NODE" != "0" ] && [ -n "$GPUS_ON_NODE" ] && NV_FLAG=(--nv)

echo "[sw-run] launching: ${RUN_INSIDE[*]}"
"$SING_BIN" exec \
    "${BIND_FLAGS[@]}" \
    "${ENV_FLAGS[@]}" \
    --env "PYTHONNOUSERSITE=1" \
    "${NV_FLAG[@]}" --containall "$SIF_LOCAL" \
    "${RUN_INSIDE[@]}"
# Capture the real exit code BEFORE cleanup (each rm/rsync would overwrite $?).
# submit_slurm.sh's crash classifier only runs when this wrapper exits non-zero.
JOB_EXIT_CODE=$?

# --- Cache writeback -----------------------------------------------------------------
if [ -n "$CACHE_RECORDS" ]; then
    declare -A WROTE_SLUG
    while IFS='|' read -r slug container writeback mode ensure_dirs; do
        [ -z "$slug" ] && continue
        [ "$writeback" != "1" ] && continue
        [ -n "${WROTE_SLUG[$slug]:-}" ] && continue
        WROTE_SLUG[$slug]=1
        HOST_DIR="$SW_PROJECT_DIR/cache/$slug"
        LOCAL_DIR="$JOB_TMPDIR/cache/${slug//\//_}"
        EXTRA=()
        [ "$mode" = "ignore-existing" ] && EXTRA=(--ignore-existing)
        rsync -a "${EXTRA[@]}" "$LOCAL_DIR/" "$HOST_DIR/" \
            || echo "[sw-run] WARNING: cache writeback failed for $slug"
    done <<< "$CACHE_RECORDS"
fi

cleanup
echo "(run_container.sh): done (exit code: $JOB_EXIT_CODE)"
exit "$JOB_EXIT_CODE"
