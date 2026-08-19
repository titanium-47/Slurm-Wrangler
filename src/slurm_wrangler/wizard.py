"""Interactive onboarding.

  sw setup  — one-time global config: NetID, lab account, cluster storage roots,
              ssh-config check. Writes ~/.config/slurm_wrangler/config.yaml.
  sw init   — per-project onboarding: detects env files, scans for large
              directories, asks a handful of questions, writes a commented sw.yaml.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .config import (GLOBAL_CONFIG_DIR, GLOBAL_CONFIG_PATH, PROJECT_CONFIG_NAME,
                     GlobalConfig, is_interactive)
from .remote import info, warn


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val or default


def ask_yn(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        val = input(f"  {prompt} [{d}]: ").strip().lower()
        if not val:
            return default
        if val in ("y", "yes"):
            return True
        if val in ("n", "no"):
            return False


def ask_choice(prompt: str, choices: List[Tuple[str, str]], default: int = 0) -> str:
    """choices: list of (value, description). Returns the chosen value."""
    print(f"  {prompt}")
    for i, (val, desc) in enumerate(choices):
        marker = "*" if i == default else " "
        print(f"    {marker} [{i + 1}] {desc}")
    while True:
        raw = input(f"  choice [1-{len(choices)}, default {default + 1}]: ").strip()
        if not raw:
            return choices[default][0]
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1][0]


# ---------------------------------------------------------------------------
# sw setup
# ---------------------------------------------------------------------------
SSH_BLOCK_TEMPLATE = """
Host {alias} {host}
    HostName {host}
    User {user}
    # Reuse one authenticated connection so Duo is answered once per session.
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist yes
"""

CLUSTER_HOSTS = {
    "hyak": ("klone-login", "klone.hyak.uw.edu"),
    "tillicum": ("tillicum", "tillicum.hyak.uw.edu"),
}


def _ssh_config_has(alias: str) -> bool:
    path = Path.home() / ".ssh" / "config"
    if not path.exists():
        return False
    pat = re.compile(rf"^\s*Host\s+.*\b{re.escape(alias)}\b", re.MULTILINE)
    return bool(pat.search(path.read_text()))


def run_setup() -> None:
    if not is_interactive():
        raise SystemExit("sw setup is interactive; run it in a real terminal.")
    print("\n=== slurm_wrangler global setup ===")
    print(f"(writes {GLOBAL_CONFIG_PATH}; safe to re-run — existing values are defaults)\n")

    existing = {}
    if GLOBAL_CONFIG_PATH.exists():
        existing = yaml.safe_load(GLOBAL_CONFIG_PATH.read_text()) or {}
    ex_clusters = existing.get("clusters", {}) or {}

    user = ask("UW NetID", existing.get("user", os.environ.get("USER", "")))

    cfg: Dict = {"user": user, "default_cluster": None, "forward_env": existing.get("forward_env", ["WANDB_*"]),
                 "clusters": {}}

    # --- ssh config ---
    for cname, (alias, host) in CLUSTER_HOSTS.items():
        if _ssh_config_has(alias):
            print(f"  ssh alias '{alias}' already in ~/.ssh/config — good.")
            continue
        if ask_yn(f"ssh alias '{alias}' not found in ~/.ssh/config. Add it (with ControlMaster for single-Duo)?"):
            block = SSH_BLOCK_TEMPLATE.format(alias=alias, host=host, user=user)
            with open(Path.home() / ".ssh" / "config", "a") as f:
                f.write(block)
            info(f"appended '{alias}' block to ~/.ssh/config")
            print(f"  -> register your key once: ssh-copy-id {alias}")

    # --- hyak ---
    if ask_yn("Configure Hyak (klone)?", True):
        prev = ex_clusters.get("hyak", {})
        account = ask("Hyak lab group / SLURM account", prev.get("slurm", {}).get("account", "weirdlab"))
        storage = ask("Hyak storage root (code/images/logs live under this)",
                      prev.get("storage_root", f"/gscratch/{account}/{user}/sw"))
        part = ask_choice("Default Hyak partition:", [
            ("ckpt", "ckpt — big idle pool, preemptible (auto-requeue handles eviction); account ckpt-" + account + ", qos ckpt-gpu"),
            (f"gpu-a40", "gpu-a40 — guaranteed lab allocation"),
            (f"gpu-l40s", "gpu-l40s — guaranteed lab allocation"),
        ])
        slurm = {"account": account, "partition": part, "constraint": "a40|a100",
                 "cpus_per_gpu": 6, "mem_mode": "per_node_from_gpu", "mem_per_gpu_gb": 60,
                 "time": "24:00:00"}
        if part == "ckpt":
            slurm["account"] = f"ckpt-{account}"
            slurm["qos"] = "ckpt-gpu"
            print(f"  (ckpt selected: account={slurm['account']}, qos=ckpt-gpu, constraint=a40|a100)")
        cfg["clusters"]["hyak"] = {"storage_root": storage, "slurm": slurm}

    # --- tillicum ---
    if ask_yn("Configure Tillicum?", True):
        prev = ex_clusters.get("tillicum", {})
        storage = ask("Tillicum storage root",
                      prev.get("storage_root", f"/gpfs/projects/stf/{user}/sw"))
        partition = ask("Default Tillicum partition (empty = cluster default)",
                        prev.get("slurm", {}).get("partition", "gpu-h200"))
        cfg["clusters"]["tillicum"] = {
            "storage_root": storage,
            "slurm": {"partition": partition, "qos": "normal", "cpus_per_gpu": 8,
                      "mem_mode": "per_cpu", "mem_per_cpu_gb": 25, "time": "24:00:00"},
        }

    if not cfg["clusters"]:
        raise SystemExit("no clusters configured; nothing to write.")
    names = list(cfg["clusters"])
    cfg["default_cluster"] = names[0] if len(names) == 1 else ask_choice(
        "Default cluster:", [(n, n) for n in names])

    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    info(f"wrote {GLOBAL_CONFIG_PATH}")

    if "WANDB_API_KEY" not in os.environ:
        warn("WANDB_API_KEY is not exported in this shell. If you log to wandb, add\n"
             '       export WANDB_API_KEY="<key>"   to your ~/.bashrc')
    print("\nNext steps:")
    print("  1. ssh once per cluster in a real terminal to warm the Duo socket, e.g.: ssh klone-login true")
    print("  2. sw doctor            # verify everything end to end")
    print("  3. cd <your project> && sw init")


# ---------------------------------------------------------------------------
# sw init
# ---------------------------------------------------------------------------
def _detect_env_files(root: Path) -> List[Tuple[str, str]]:
    """Returns (value, description) choices for the image question."""
    choices: List[Tuple[str, str]] = []
    for cand in ("environment.yml", "environment.yaml", "environment.sw.yml"):
        if (root / cand).exists():
            choices.append((f"conda:{cand}", f"conda env file: {cand}  (recommended)"))
    for cand in ("Dockerfile", "docker/Dockerfile", "docker/Dockerfile.base"):
        if (root / cand).exists():
            choices.append((f"dockerfile:{cand}", f"project Dockerfile: {cand}"))
    if (root / "requirements.txt").exists():
        choices.append(("requirements", "requirements.txt (wrapped into a generated environment.yml)"))
    choices.append(("export", "export a local conda env to environment.sw.yml now"))
    choices.append(("skip", "skip for now (fill in image: later)"))
    return choices


def _conda_env_names() -> List[str]:
    try:
        out = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split()[0]
        if name not in ("*",):
            names.append(name)
    return names


def _scan_top_level(root: Path, threshold_gb: float = 1.0) -> List[Tuple[str, float]]:
    """Quick raw du of depth-1 entries (pre-filter — just to inform the wizard)."""
    big = []
    for child in sorted(root.iterdir()):
        if child.name in (".git", ".sw"):
            continue
        try:
            if child.is_dir():
                out = subprocess.run(["du", "-sb", str(child)], capture_output=True,
                                     text=True, timeout=120).stdout
                nbytes = int(out.split()[0]) if out else 0
            else:
                nbytes = child.stat().st_size
        except Exception:
            continue
        gb = nbytes / (1 << 30)
        if gb >= threshold_gb:
            big.append((child.name + ("/" if child.is_dir() else ""), gb))
    return big


SW_YAML_TEMPLATE = """\
# slurm_wrangler project config — see https://github.com/{gh_user}/slurm_wrangler
# Run jobs with:   sw run [--cluster hyak --gpus 1 ...] -- python train.py ...
name: {name}

# Container image, content-hashed: changing this section (or the env file it
# points at) triggers exactly one rebuild+push; code changes never do.
image:
{image_block}

# What gets rsynced to the cluster per job (content-addressed snapshots).
# .gitignore'd and .dockerignore'd files are skipped automatically.
sync:
  respect_gitignore: true
  respect_dockerignore: true
  exclude:{exclude_block}

# Datasets: pushed once with `sw data push <dir>`, mounted read-only at `mount`.
data:{data_block}

# Project-relative dirs backed by persistent cluster storage (survive the job,
# shared across jobs — checkpoints/logs belong here).
persist: [logs, outputs]

# Caches staged to node-local scratch and bound into the container.
# Presets: huggingface, torch, pip, isaac-sim. Example:
#   cache:
#     - preset: huggingface
cache: []

# plain: run the command as-is. torchrun: wrap in torch.distributed.run
# (use for multi-GPU/multi-node DDP; command becomes `train.py ...`).
launcher: {launcher}

defaults:
  cluster: {cluster}
  gpus: {gpus}
  time: "{time}"
"""


def run_init(root: Optional[Path] = None) -> None:
    if not is_interactive():
        raise SystemExit("sw init is interactive; run it in a real terminal.")
    root = (root or Path.cwd()).resolve()
    cfg_path = root / PROJECT_CONFIG_NAME
    if cfg_path.exists():
        if not ask_yn(f"{PROJECT_CONFIG_NAME} already exists here. Overwrite?", False):
            raise SystemExit("aborted.")

    gcfg = GlobalConfig.load()
    if not gcfg.clusters:
        warn("no clusters configured yet — run `sw setup` first (continuing anyway).")

    print(f"\n=== sw init: onboarding {root} ===\n")
    default_name = re.sub(r"[^A-Za-z0-9_-]", "-", root.name)
    name = ask("Project name", default_name)

    # --- image ---
    print("\nHow should the cluster environment be built?")
    choice = ask_choice("Environment source:", _detect_env_files(root))
    image_block = "  kind: conda\n  environment: environment.yml"
    if choice.startswith("conda:"):
        envfile = choice.split(":", 1)[1]
        image_block = f"  kind: conda\n  environment: {envfile}"
    elif choice.startswith("dockerfile:"):
        df = choice.split(":", 1)[1]
        image_block = f"  kind: dockerfile\n  dockerfile: {df}"
    elif choice == "requirements":
        envfile = "environment.sw.yml"
        py = ask("Python version for the generated env", "3.10")
        (root / envfile).write_text(
            f"name: {name}\nchannels: [conda-forge]\ndependencies:\n"
            f"  - python={py}\n  - pip\n  - pip:\n      - -r requirements.txt\n"
        )
        info(f"wrote {envfile} (wraps requirements.txt)")
        image_block = f"  kind: conda\n  environment: {envfile}"
        warn("note: requirements.txt itself is hashed only via the env file — "
             "bump a comment in environment.sw.yml after changing requirements.txt, "
             "or list deps directly in the env file.")
    elif choice == "export":
        envs = _conda_env_names()
        env_name = ask(f"Local conda env to export ({', '.join(envs[:8])}...)" if envs
                       else "Local conda env to export",
                       os.environ.get("CONDA_DEFAULT_ENV", ""))
        envfile = "environment.sw.yml"
        info(f"running: conda env export -n {env_name} --no-builds > {envfile}")
        with open(root / envfile, "w") as f:
            subprocess.run(["conda", "env", "export", "-n", env_name, "--no-builds"],
                           stdout=f, check=True)
        # strip the local-only `prefix:` line
        txt = (root / envfile).read_text()
        txt = re.sub(r"^prefix:.*\n?", "", txt, flags=re.MULTILINE)
        (root / envfile).write_text(txt)
        info(f"wrote {envfile}")
        warn("full env exports often contain machine-specific pins that fail to solve "
             "on the first build — trimming it down to the packages you import is more robust.")
        image_block = f"  kind: conda\n  environment: {envfile}"

    # --- big directories ---
    print("\nScanning top-level entries over 1 GB (raw sizes, pre-ignore-filters)...")
    big = _scan_top_level(root)
    excludes: List[str] = []
    datasets: List[Tuple[str, str]] = []
    if big:
        for entry, gb in big:
            print(f"\n  {entry}  ({gb:.1f} GB)")
            action = ask_choice("What should happen to it?", [
                ("exclude", "exclude from sync (default for outputs/checkpoints)"),
                ("data", "treat as dataset: exclude from sync, push once with `sw data push`, mount read-only"),
                ("sync", "sync it with the code every time the content changes"),
                ("ignore", "decide later (the >1GB gate will ask at first `sw run`)"),
            ])
            clean = entry.rstrip("/")
            if action == "exclude":
                excludes.append(f"/{clean}")
            elif action == "data":
                excludes.append(f"/{clean}")
                datasets.append((clean, f"/data/{clean}"))
    else:
        print("  none found.")

    # --- launcher + defaults ---
    print()
    launcher = ask_choice("Launcher:", [
        ("plain", "plain — run the command as-is (single-node, most jobs)"),
        ("torchrun", "torchrun — torch.distributed DDP wrapper (multi-GPU/node)"),
    ])
    cluster_names = list(gcfg.clusters) or ["hyak"]
    cluster = ask_choice("Default cluster:", [(c, c) for c in cluster_names],
                         default=cluster_names.index(gcfg.default_cluster)
                         if gcfg.default_cluster in cluster_names else 0)
    gpus = ask("Default GPUs per job", "1")
    time_lim = ask("Default time limit", "24:00:00")

    exclude_block = "" if not excludes else "\n" + "\n".join(f"    - \"{e}\"" for e in excludes)
    exclude_block = exclude_block or " []"
    if exclude_block == " []":
        exclude_block = " []"
    data_block = "" if not datasets else "\n" + "\n".join(
        f"  - name: {n}\n    src: {root / n}\n    mount: {m}" for n, m in datasets)
    data_block = data_block or " []"

    content = SW_YAML_TEMPLATE.format(
        gh_user="yandabao", name=name, image_block=image_block,
        exclude_block=exclude_block, data_block=data_block,
        launcher=launcher, cluster=cluster, gpus=gpus, time=time_lim,
    )
    cfg_path.write_text(content)
    info(f"wrote {cfg_path}")

    # .gitignore hygiene
    gi = root / ".gitignore"
    if gi.exists() and ".sw/" not in gi.read_text():
        if ask_yn("Add .sw/ (local slurm_wrangler state) to .gitignore?", True):
            with open(gi, "a") as f:
                f.write("\n# slurm_wrangler local state\n.sw/\n")
    print("\nNext steps:")
    for n, _ in datasets:
        print(f"  sw data push {n}")
    print("  sw doctor               # verify cluster reachability + env file")
    print("  sw env push             # build + upload the image (once per dep change)")
    print("  sw run -- python <script.py> ...   # or add e.g. --cluster tillicum --gpus 4")
