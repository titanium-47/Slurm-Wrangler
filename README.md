# slurm_wrangler (`sw`)

Run **any local project** on UW's **Hyak (klone)** and **Tillicum** SLURM clusters with one command:

```bash
cd ~/my_project
sw run --gpus 1 -- python train.py --epochs 100
```

That single command:

1. **Syncs your code** to the cluster — respecting `.gitignore` / `.dockerignore`, as a
   **content-addressed snapshot** (identical code is never shipped twice; a 20-job sweep
   syncs once).
2. **Ensures the right container env** is on the cluster — your `environment.yml` is
   content-hashed into an image tag; changing a dependency triggers exactly one automatic
   rebuild+push, and code edits never do.
3. **Submits a SLURM job** that runs your command inside that container (Apptainer), with
   GPUs, your `WANDB_*` env forwarded, and battle-tested resilience machinery:
   auto-requeue on preemption/time-limit, a crash classifier that retries transient GPU
   faults (ECC) on a fresh node but fails fast on real bugs, and a shared bad-node
   registry that keeps jobs off persistently broken nodes.

The resilience layer is extracted from UWLab's cluster infra (`submit_job_slurm_hyak.sh`
et al.), generalized so any project can use it.

## Install

```bash
git clone git@github.com:yandabao/slurm_wrangler.git
cd slurm_wrangler
pipx install -e .        # or: pip install --user -e .
```

Local requirements: `rsync`, `ssh`, and — only for building images — `docker` + `apptainer`.

## One-time setup

```bash
sw setup      # interactive: NetID, lab account, storage roots, ssh-config blocks
ssh klone-login true   # answer Duo ONCE in a real terminal; ControlPersist keeps it warm
sw doctor     # end-to-end preflight: ssh, storage, sbatch, apptainer, env file
```

`sw setup` writes `~/.config/slurm_wrangler/config.yaml` and can append the ssh aliases
(`klone-login`, `tillicum`) with `ControlMaster`/`ControlPersist` so **Duo is answered at
most once per session**, no matter how many jobs you launch.

## Per-project onboarding

```bash
cd ~/my_project
sw init       # interactive wizard
```

The wizard detects your `environment.yml` / `Dockerfile` (or exports your current conda
env), scans for >1 GB top-level directories and asks what to do with each (exclude /
dataset / sync), and writes a commented `sw.yaml`. Example:

```yaml
name: my-project

image:
  kind: conda                    # conda | dockerfile | prebuilt
  environment: environment.yml   # the SAME file you use locally
  # apt: [ffmpeg]                # extra system packages
  # pip: ["some-pkg==1.2"]       # extras installed after the env

sync:
  respect_gitignore: true
  respect_dockerignore: true
  exclude: ["/checkpoints"]      # extra gitignore-style patterns

data:                            # pushed ONCE with `sw data push`, mounted read-only
  - name: language_table
    src: ~/data/language_table
    mount: /data/language_table

persist: [logs, outputs]         # project dirs backed by persistent cluster storage

cache: []                        # node-local staged caches; presets: huggingface,
                                 # torch, pip, isaac-sim

launcher: plain                  # plain | torchrun (DDP wrapper for multi-GPU/node)

defaults:
  cluster: hyak
  gpus: 1
  time: "24:00:00"
```

## Everyday commands

```bash
sw run --gpus 2 --time 8:00:00 -- python train.py lr=1e-4     # submit
sw run --cluster tillicum --gpus 4 -- python train.py         # other cluster
sw run --dry-run -- python train.py                           # show everything, submit nothing
sw status                     # squeue + this project's recent submissions
sw logs 12345 -f              # tail the job log live (survives requeues)
sw cancel 12345               # scancel (disables auto-requeue first)
sw env status|build|push      # image lifecycle (usually automatic)
sw data push ~/data/foo       # ship a dataset once
sw sync --review              # revisit large-directory decisions
sw prune --keep 5             # delete old code snapshots on the cluster
sw doctor                     # preflight checks
```

Everything after `--` is your command, run verbatim inside the container with the synced
project as the working directory.

## How it works

```
 local                          login node                     compute node
 ─────                          ──────────                     ────────────
 sw run
  ├─ hash env spec ────────────▶ images/env-<hash8>.sif        (built+pushed only when
  ├─ enumerate + gate files      code/<hash12>/                 the hash is new)
  ├─ hash code, rsync snapshot ▶
  └─ ssh: submit_slurm.sh ─────▶ bad-node registry → --exclude
                                 generate job.sh, sbatch ─────▶ run_container.sh
                                                                ├─ stage .sif + code + caches
                                                                │  to node-local scratch
                                                                ├─ bind persist/data/caches
                                                                └─ apptainer exec <your cmd>
                                                                requeue traps + crash classifier
```

Remote layout (per cluster, under your storage root):

```
sw/
├── runtime/<hash>/              # sw's own scripts, content-addressed
└── projects/<name>/
    ├── code/<hash12>/           # immutable code snapshots (deduplicated)
    ├── images/env-<hash8>.sif   # container images, keyed by env hash
    ├── data/<dataset>/          # pushed once, mounted read-only
    ├── persist/logs, outputs/   # survive jobs; bound into the container
    ├── cache/<slug>/            # persistent caches (staged node-local per job)
    ├── slurm_logs/              # SLURM .out/.err + bad_nodes.tsv
    └── tmp/                     # persistent /tmp for jobs
```

### The environment story

Your conda env is **not** shipped (glibc mismatch between Ubuntu and the clusters' Rocky
Linux makes conda-pack a footgun). Instead, the same `environment.yml` you use locally is
built into a micromamba container image, converted to a `.sif`, and uploaded — keyed by
content hash. One source of truth; the cluster env can't drift from the file.

- `image.kind: dockerfile` — full control for complex images (Isaac Sim, custom CUDA).
- `image.kind: prebuilt` — point at an existing `.sif` on the cluster; no local docker needed.
- Jobs run with `micromamba run -p /opt/env` so conda activation scripts
  (`CUDA_HOME` etc.) apply, and with `--containall` for a clean, reproducible env.

### The >1 GB gate

At sync time, any **top-level** directory (or root file) over 1 GB / 50k files that
survives the ignore filters triggers a one-time question: sync it, exclude it, or treat
it as a dataset. The answer is remembered in `.sw/sync.local.yaml` (re-asked only if it
doubles in size; review with `sw sync --review`). A "yes" covers the whole directory —
no per-subdirectory nagging. Non-interactive runs fail fast with exact instructions
instead of hanging.

### Failure handling (from UWLab, generalized)

| Event | What happens |
|---|---|
| Preemption (ckpt) / time limit | USR1/TERM trap → `scontrol requeue`; logs append across attempts |
| Transient GPU fault (ECC, Xid, NVLink…) | Faulting node recorded in `bad_nodes.tsv`; auto-requeue (≤5) on a fresh node |
| Deterministic bug (ImportError, OOM, SyntaxError…) | **No requeue** — fails fast, even if a hardware signature co-occurs |
| Scheduled onto a known-bad node | Bounce guard requeues within seconds, before any staging |
| `sw cancel` | Sets `Requeue=0` first so the classifier honors the operator cancel |

Registry entries expire after 7 days (nodes get repaired) and the exclude list is capped
at 30 nodes, so it can never make your jobs unschedulable. Knobs: `--exclude-nodes`,
`--skip-bad-nodes`, `--no-requeue`.

**Requeue ≠ resume:** a requeued job restarts your command from the beginning. Make your
training script resume from the latest checkpoint in a `persist:` dir (e.g. `logs/`) —
then requeues continue seamlessly.

### Notes & limitations

- **Writes outside `persist:` dirs vanish** when the job ends — code runs from a node-local
  scratch copy. Put checkpoints/logs in a persist dir.
- Multi-node jobs need `launcher: torchrun` (with `plain`, every node runs the same command).
- `.dockerignore` negation handling is a pragmatic subset; prefer simple patterns.
- Hyak ckpt: use `account: ckpt-<lab>`, `qos: ckpt-gpu` (the wizard sets this). The GPU
  constraint is auto-forced to `a40|a100` on Hyak — override with `--constraint`.
- Channels (long-lived GPU-holder workers with instant push, as in UWLab) are not ported
  yet; direct `sbatch` submission is the supported path.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e . pytest
.venv/bin/python -m pytest tests/ -q
```

Runtime scripts (`src/slurm_wrangler/runtime/*.sh`) are shipped to the cluster
content-addressed, so in-flight jobs keep the exact runtime they were submitted with.
