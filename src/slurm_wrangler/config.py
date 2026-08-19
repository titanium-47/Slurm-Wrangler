"""Configuration model + layered resolution.

Three layers, later wins:
  1. Built-in defaults (this file)
  2. Global user config      ~/.config/slurm_wrangler/config.yaml   (netid, clusters)
  3. Project config          <project>/sw.yaml                      (image, sync, defaults)
  4. CLI flags               (handled by callers via SubmitParams overrides)

Per-project mutable state (sync decisions, job history, hash cache) lives in
<project>/.sw/ and is git-ignored.
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

GLOBAL_CONFIG_DIR = Path(os.environ.get("SW_CONFIG_DIR", "~/.config/slurm_wrangler")).expanduser()
GLOBAL_CONFIG_PATH = GLOBAL_CONFIG_DIR / "config.yaml"
PROJECT_CONFIG_NAME = "sw.yaml"
STATE_DIR_NAME = ".sw"

# ---------------------------------------------------------------------------
# Built-in cluster knowledge (UW-specific facts live here, overridable in the
# global config). `storage_root` is per-user and comes from `sw setup`.
# ---------------------------------------------------------------------------
BUILTIN_CLUSTERS: Dict[str, Dict[str, Any]] = {
    "hyak": {
        "login": "klone-login",
        "mount_dir": "/mmfs1",
        "scheduler": "slurm",
        # Hyak (klone) ground truth: the weirdlab allocation can only schedule
        # a40/a100 nodes; requesting anything else is rejected at submit time.
        # a40 nodes fault often (uncorrectable ECC) but the auto-requeue +
        # bad-node registry make that survivable.
        "force_constraint": "a40|a100",
        "requeue_signal_secs": 30,
        "slurm": {
            "account": "weirdlab",
            "partition": "ckpt",
            "qos": "",  # ckpt GPU jobs usually need qos ckpt-gpu; wizard sets this
            "constraint": "a40|a100",
            "cpus_per_gpu": 6,
            "mem_mode": "per_node_from_gpu",   # mem = gpus * mem_per_gpu
            "mem_per_gpu_gb": 60,
            "mem_cpu_job_gb": 32,              # per-node mem when --gpus 0
            "time": "24:00:00",
        },
    },
    "tillicum": {
        "login": "tillicum",
        "mount_dir": "/gpfs",
        "scheduler": "slurm",
        "force_constraint": "",
        "requeue_signal_secs": 120,
        "slurm": {
            "account": "",
            "partition": "",
            "qos": "normal",
            "constraint": "",
            "cpus_per_gpu": 8,
            "mem_mode": "per_cpu",             # tillicum convention: mem-per-cpu
            "mem_per_cpu_gb": 25,
            "mem_cpu_job_gb": 32,
            "time": "24:00:00",
        },
    },
}

DEFAULT_FORWARD_ENV = ["WANDB_*"]

# Sync gate thresholds
BIG_DIR_BYTES = 1 << 30          # 1 GiB per top-level dir (or root file) triggers a prompt
BIG_DIR_FILES = 50_000           # ...or this many files
REASK_GROWTH_FACTOR = 2.0        # re-prompt if an approved dir grew by this factor


class ConfigError(Exception):
    pass


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
@dataclass
class ClusterConfig:
    name: str
    login: str
    storage_root: str            # absolute remote path, e.g. /gscratch/weirdlab/yandabao/sw
    mount_dir: str
    scheduler: str = "slurm"
    force_constraint: str = ""
    requeue_signal_secs: int = 60
    slurm: Dict[str, Any] = field(default_factory=dict)

    def project_dir(self, project: str) -> str:
        return f"{self.storage_root}/projects/{project}"

    def runtime_dir(self, rt_hash: str) -> str:
        return f"{self.storage_root}/runtime/{rt_hash}"

    def slurm_logs_dir(self, project: str) -> str:
        return f"{self.project_dir(project)}/slurm_logs"


@dataclass
class GlobalConfig:
    user: str = ""
    forward_env: List[str] = field(default_factory=lambda: list(DEFAULT_FORWARD_ENV))
    default_cluster: str = "hyak"
    clusters: Dict[str, ClusterConfig] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "GlobalConfig":
        raw = _load_yaml(GLOBAL_CONFIG_PATH)
        merged_clusters: Dict[str, ClusterConfig] = {}
        raw_clusters = raw.get("clusters", {}) or {}
        names = set(BUILTIN_CLUSTERS) | set(raw_clusters)
        for name in names:
            spec = _deep_merge(BUILTIN_CLUSTERS.get(name, {}), raw_clusters.get(name, {}) or {})
            if "storage_root" not in spec:
                continue  # unconfigured builtin cluster: skip until `sw setup`
            merged_clusters[name] = ClusterConfig(
                name=name,
                login=spec.get("login", name),
                storage_root=str(spec["storage_root"]).rstrip("/"),
                mount_dir=spec.get("mount_dir", "/"),
                scheduler=spec.get("scheduler", "slurm"),
                force_constraint=spec.get("force_constraint", ""),
                requeue_signal_secs=int(spec.get("requeue_signal_secs", 60)),
                slurm=spec.get("slurm", {}) or {},
            )
        return cls(
            user=raw.get("user", ""),
            forward_env=raw.get("forward_env", list(DEFAULT_FORWARD_ENV)),
            default_cluster=raw.get("default_cluster", "hyak"),
            clusters=merged_clusters,
        )

    def cluster(self, name: Optional[str]) -> ClusterConfig:
        name = name or self.default_cluster
        if name not in self.clusters:
            configured = ", ".join(sorted(self.clusters)) or "(none)"
            raise ConfigError(
                f"Cluster '{name}' is not configured. Configured: {configured}.\n"
                f"Run `sw setup` to configure clusters (writes {GLOBAL_CONFIG_PATH})."
            )
        return self.clusters[name]


# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------
@dataclass
class ImageConfig:
    kind: str = "conda"                  # conda | dockerfile | prebuilt
    environment: str = "environment.yml" # conda kind: env file, relative to project root
    base: str = "mambaorg/micromamba:2.0.5"
    apt: List[str] = field(default_factory=list)
    pip: List[str] = field(default_factory=list)
    dockerfile: str = ""                 # dockerfile kind
    build_args: Dict[str, str] = field(default_factory=dict)
    sif_remote: str = ""                 # prebuilt kind: absolute path on the cluster


@dataclass
class SyncConfig:
    respect_gitignore: bool = True
    respect_dockerignore: bool = True
    exclude: List[str] = field(default_factory=list)
    follow_symlinks: bool = False
    big_dir_gb: float = 1.0
    big_dir_files: int = BIG_DIR_FILES


@dataclass
class DataConfig:
    name: str
    src: str = ""                        # local dir for `sw data push`
    mount: str = ""                      # container mount point; default /data/<name>


@dataclass
class CacheConfig:
    container: str                       # container path, e.g. /root/.cache/huggingface
    host_slug: str = ""                  # persistent dir name; derived from container path
    writeback: bool = False
    writeback_mode: str = "merge"        # merge | ignore-existing
    ensure_dirs: List[str] = field(default_factory=list)


@dataclass
class ProjectConfig:
    root: Path
    name: str
    image: ImageConfig
    sync: SyncConfig
    data: List[DataConfig]
    cache: List[CacheConfig]
    persist: List[str]
    env: Dict[str, str]
    forward_env: List[str]
    launcher: str                        # plain | torchrun
    workdir: str
    prestage: str                        # optional project script run on the compute node pre-container
    defaults: Dict[str, Any]

    @property
    def state_dir(self) -> Path:
        d = self.root / STATE_DIR_NAME
        d.mkdir(exist_ok=True)
        return d

    @classmethod
    def load(cls, start: Optional[Path] = None) -> "ProjectConfig":
        root = find_project_root(start)
        if root is None:
            raise ConfigError(
                f"No {PROJECT_CONFIG_NAME} found here or in any parent directory.\n"
                f"Run `sw init` in your project root to create one."
            )
        raw = _load_yaml(root / PROJECT_CONFIG_NAME)
        return cls.from_dict(root, raw)

    @classmethod
    def from_dict(cls, root: Path, raw: Dict[str, Any]) -> "ProjectConfig":
        name = raw.get("name") or root.name
        if not str(name).replace("-", "").replace("_", "").isalnum():
            raise ConfigError(f"Project name '{name}' must be alphanumeric plus -/_")

        img_raw = raw.get("image", {}) or {}
        image = ImageConfig(
            kind=img_raw.get("kind", "conda"),
            environment=img_raw.get("environment", "environment.yml"),
            base=img_raw.get("base", ImageConfig.base),
            apt=list(img_raw.get("apt", []) or []),
            pip=list(img_raw.get("pip", []) or []),
            dockerfile=img_raw.get("dockerfile", ""),
            build_args={str(k): str(v) for k, v in (img_raw.get("build_args", {}) or {}).items()},
            sif_remote=img_raw.get("sif_remote", ""),
        )
        if image.kind not in ("conda", "dockerfile", "prebuilt"):
            raise ConfigError(f"image.kind must be conda|dockerfile|prebuilt, got '{image.kind}'")

        sync_raw = raw.get("sync", {}) or {}
        sync = SyncConfig(
            respect_gitignore=bool(sync_raw.get("respect_gitignore", True)),
            respect_dockerignore=bool(sync_raw.get("respect_dockerignore", True)),
            exclude=list(sync_raw.get("exclude", []) or []),
            follow_symlinks=bool(sync_raw.get("follow_symlinks", False)),
            big_dir_gb=float(sync_raw.get("big_dir_gb", 1.0)),
            big_dir_files=int(sync_raw.get("big_dir_files", BIG_DIR_FILES)),
        )

        data = []
        for d in raw.get("data", []) or []:
            if isinstance(d, str):
                d = {"name": d}
            name_ = d.get("name") or (Path(d.get("src", "")).name if d.get("src") else "")
            if not name_:
                raise ConfigError(f"data entry needs a name: {d}")
            data.append(DataConfig(
                name=name_,
                src=d.get("src", ""),
                mount=d.get("mount", f"/data/{name_}"),
            ))

        from .presets import expand_cache_entries  # local import to avoid cycle
        cache = expand_cache_entries(raw.get("cache", []) or [])

        launcher = raw.get("launcher", "plain")
        if launcher not in ("plain", "torchrun"):
            raise ConfigError(f"launcher must be plain|torchrun, got '{launcher}'")

        workdir = raw.get("workdir", "") or f"/workspace/{name}"

        return cls(
            root=root,
            name=str(name),
            image=image,
            sync=sync,
            data=data,
            cache=cache,
            persist=list(raw.get("persist", ["logs", "outputs"]) or []),
            env={str(k): str(v) for k, v in (raw.get("env", {}) or {}).items()},
            forward_env=list(raw.get("forward_env", []) or []),
            launcher=launcher,
            workdir=workdir,
            prestage=raw.get("prestage", "") or "",
            defaults=raw.get("defaults", {}) or {},
        )


def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / PROJECT_CONFIG_NAME).exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Per-project local state helpers (.sw/)
# ---------------------------------------------------------------------------
def load_state(project: ProjectConfig, filename: str) -> Dict[str, Any]:
    return _load_yaml(project.state_dir / filename)


def save_state(project: ProjectConfig, filename: str, data: Dict[str, Any]) -> None:
    path = project.state_dir / filename
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=True)


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()
