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

## Install

```bash
git clone git@github.com:yandabao/slurm_wrangler.git
cd slurm_wrangler
pipx install -e .        # or: pip install --user -e .
```

Local requirements: `rsync`, `ssh`, and — only for building images — `docker` + `apptainer`.

## One-time setup

### 1. SSH: keys + a persistent connection (do this first)

`sw` reaches the clusters through the ssh aliases `klone-login` (Hyak) and `tillicum`.
Hyak enforces Duo 2FA, and a single `sw run` opens several ssh/rsync connections — so
the aliases **must** use `ControlMaster`/`ControlPersist`. With them, one authenticated
connection is reused: you answer Duo **once**, and every later `sw` command (including
ones run from scripts or agents) rides the open socket with no prompt.

Add this to `~/.ssh/config` (replace `<netid>`):

```sshconfig
Host klone-login klone klone.hyak.uw.edu
    HostName klone.hyak.uw.edu
    User <netid>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist yes

Host tillicum tillicum.hyak.uw.edu
    HostName tillicum.hyak.uw.edu
    User <netid>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist yes
```

`ControlPersist yes` keeps the master connection open indefinitely after your first
login (use e.g. `ControlPersist 4h` if you'd rather it expire). Then register a key and
open the master connection once, **in a real terminal** (Duo needs a TTY):

```bash
ssh-keygen -t ed25519            # if you don't already have a key
ssh-copy-id klone-login          # register it on Hyak (Duo prompt)
ssh-copy-id tillicum             # and Tillicum
ssh klone-login true             # authenticate once -> master socket opens
ssh -O check klone-login         # should print "Master running"
```

If a socket ever dies (reboot, network change), just run `ssh klone-login true` again.
`sw setup` will offer to append these blocks for you if they're missing.

### 2. Configure and verify

```bash
sw setup      # interactive: NetID, lab account, storage roots (offers the ssh blocks above)
sw doctor     # end-to-end preflight: ssh sockets, storage, sbatch, apptainer, env file
```

`sw setup` writes `~/.config/slurm_wrangler/config.yaml`. `sw doctor` warns if a
cluster has no live master socket, since the next command would trigger a Duo prompt.

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

### Portable `sw.yaml` (placeholders and vars)

To commit one `sw.yaml` for a whole team, keep usernames and per-user paths out of it.
`image.sif_remote` and `env:` values are expanded at submit time:

| Placeholder | Expands to |
|---|---|
| `{storage_root}` | your cluster `storage_root` from `~/.config/slurm_wrangler/config.yaml` |
| `{project_dir}` | `{storage_root}/projects/<name>` (where sw keeps persist/logs/caches) |
| `{project}` / `{cluster}` / `{user}` (`{netid}`) | project name, cluster name, global `user` |
| `{<var>}` | a var from `vars:` (below) |
| `{container:<name>}` | the same path **as seen inside the container**. Only the cluster mount (`mount_dir`) is bound, so on Hyak `/gscratch/...` becomes `/mmfs1/gscratch/...`. Use this in `env:` values that the job reads. Override with `container_storage_root:` on a cluster if your layout differs. |

`vars:` in `sw.yaml` are project **defaults**. Each user can override any of them in their own
global config, either top-level (`vars:`) or per cluster (`clusters.<name>.vars:`, which wins).
That lets one user point at an image or env that lives in **another user's** directory:

```yaml
# sw.yaml (committed)
vars:
  image_dir: "{storage_root}/apptainer"          # default: your own storage
  venv: "{container:project_dir}/env/.venv"
image:
  kind: prebuilt
  sif_remote: "{image_dir}/isaac-lab.sif"
env:
  VIRTUAL_ENV: "{venv}"
  PATH: "{venv}/bin:/usr/local/bin:/usr/bin:/bin"
```

```yaml
# ~/.config/slurm_wrangler/config.yaml (per user, not committed)
clusters:
  hyak:
    storage_root: /gscratch/weirdlab/alice
    vars:
      image_dir: /gscratch/weirdlab/bob/apptainer   # reuse Bob's copy of the image
```

Vars may reference built-ins and other vars. Braces that don't name a known placeholder,
such as JSON, are left as-is. Paths from other users' directories just need to be readable by
you.

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

### Failure handling

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
- **`HOME` inside the container is `/sw_home`** — a job-private, node-local dir discarded
  after the job (prevents dotfile-cache writes/races on GPFS). Caches worth keeping
  across jobs belong in `cache:` entries, which override the relevant paths.
- Multi-node jobs need `launcher: torchrun` (with `plain`, every node runs the same command).
- `.dockerignore` negation handling is a pragmatic subset; prefer simple patterns.
- Hyak ckpt: use `account: ckpt-<lab>`, `qos: ckpt-gpu` (the wizard sets this). The GPU
  constraint is auto-forced to `a40|a100` on Hyak — override with `--constraint`.
- Long-lived GPU-holder workers ("channels") with zero-queue-wait experiment push are
  planned; direct `sbatch` submission is the current path.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e . pytest
.venv/bin/python -m pytest tests/ -q
```

Runtime scripts (`src/slurm_wrangler/runtime/*.sh`) are shipped to the cluster
content-addressed, so in-flight jobs keep the exact runtime they were submitted with.
