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
import dataclasses
import os
import re
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
    # storage_root as seen INSIDE the container. Only mount_dir is bound, so a storage_root
    # reached through a host symlink (Hyak: /gscratch -> /mmfs1/gscratch) needs mapping.
    # Empty = derive via container_path().
    container_storage_root: str = ""
    vars: Dict[str, str] = field(default_factory=dict)  # per-user sw.yaml placeholder overrides

    def project_dir(self, project: str) -> str:
        return f"{self.storage_root}/projects/{project}"

    def container_path(self, host_path: str) -> str:
        """Map a host path to the path visible inside the container.

        Explicit ``container_storage_root`` wins. Otherwise a path already under
        ``mount_dir`` is unchanged, and anything else is assumed reachable as
        ``mount_dir + path`` (the Hyak /gscratch -> /mmfs1/gscratch convention).
        """
        if self.container_storage_root and (
            host_path == self.storage_root or host_path.startswith(self.storage_root + "/")
        ):
            return self.container_storage_root.rstrip("/") + host_path[len(self.storage_root):]
        mount = self.mount_dir.rstrip("/")
        if not host_path.startswith("/") or not mount or host_path == mount or host_path.startswith(mount + "/"):
            return host_path
        return mount + host_path

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
    vars: Dict[str, str] = field(default_factory=dict)  # per-user sw.yaml placeholder overrides

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
                container_storage_root=str(spec.get("container_storage_root", "") or ""),
                vars=_check_vars(spec.get("vars"), f"{GLOBAL_CONFIG_PATH} clusters.{name}.vars"),
            )
        return cls(
            user=raw.get("user", ""),
            forward_env=raw.get("forward_env", list(DEFAULT_FORWARD_ENV)),
            default_cluster=raw.get("default_cluster", "hyak"),
            clusters=merged_clusters,
            vars=_check_vars(raw.get("vars"), f"{GLOBAL_CONFIG_PATH} vars"),
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
# sw.yaml placeholders
# ---------------------------------------------------------------------------
# `{name}` tokens in image.sif_remote and env values are expanded at submit time, so
# one committed sw.yaml works for every user. Names are the built-ins below plus
# user-defined `vars:`. Precedence, low -> high: sw.yaml `vars:` (project defaults),
# global config `vars:`, global config `clusters.<name>.vars:`. So each user can point a
# var somewhere else, e.g. at an image that lives in another user's directory.
# `{container:name}` gives the same path as seen INSIDE the container (see
# ClusterConfig.container_path). Braces that don't name a known var are left untouched.
BUILTIN_PLACEHOLDERS = ("storage_root", "project_dir", "project", "cluster", "user", "netid")
_PLACEHOLDER_RE = re.compile(r"\{(container:)?([A-Za-z_][A-Za-z0-9_]*)\}")
_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_vars(vars_: Optional[Dict[str, Any]], where: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (vars_ or {}).items():
        k = str(k)
        if not _VAR_NAME_RE.match(k):
            raise ConfigError(f"{where}: var name '{k}' must be an identifier")
        if k in BUILTIN_PLACEHOLDERS:
            raise ConfigError(f"{where}: var '{k}' shadows a built-in placeholder")
        out[k] = str(v)
    return out


class PlaceholderResolver:
    """Expands `{name}` / `{container:name}`; vars may reference other vars and built-ins."""

    def __init__(self, project_name: str, project_vars: Dict[str, str],
                 cluster: "ClusterConfig", gcfg: "GlobalConfig"):
        self.cluster = cluster
        self.values: Dict[str, str] = {
            "storage_root": cluster.storage_root,
            "project_dir": cluster.project_dir(project_name),
            "project": project_name,
            "cluster": cluster.name,
            "user": gcfg.user,
            "netid": gcfg.user,
        }
        self.raw_vars: Dict[str, str] = {**project_vars, **gcfg.vars, **cluster.vars}
        self._resolving: List[str] = []

    def value(self, name: str) -> Optional[str]:
        if name in self.values:
            return self.values[name]
        if name not in self.raw_vars:
            return None
        if name in self._resolving:
            raise ConfigError("sw.yaml vars form a cycle: " + " -> ".join(self._resolving + [name]))
        self._resolving.append(name)
        try:
            self.values[name] = self.expand(self.raw_vars[name])
        finally:
            self._resolving.pop()
        return self.values[name]

    def expand(self, text: str) -> str:
        def sub(m: "re.Match[str]") -> str:
            to_container, name = m.group(1), m.group(2)
            val = self.value(name)
            if val is None:
                if to_container:
                    raise ConfigError(f"unknown placeholder {m.group(0)} in sw.yaml")
                return m.group(0)  # not ours (e.g. JSON in an env value); leave it
            if not val:
                hint = "user" if name in ("user", "netid") else f"vars.{name}"
                raise ConfigError(f"sw.yaml placeholder {m.group(0)} is empty "
                                  f"(set `{hint}` in {GLOBAL_CONFIG_PATH})")
            return self.cluster.container_path(val) if to_container else val
        return _PLACEHOLDER_RE.sub(sub, text)


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
class ProvenanceConfig:
    git: str = "auto"                    # auto | off
    remote: str = "origin"               # used when the current branch has no upstream
    branch_prefix: str = "autopush/"


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
    vars: Dict[str, str] = field(default_factory=dict)  # placeholder defaults; users override globally
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)

    def resolved(self, cluster: "ClusterConfig", gcfg: "GlobalConfig") -> "ProjectConfig":
        """Copy with sw.yaml placeholders expanded for ``cluster`` (see PlaceholderResolver)."""
        r = PlaceholderResolver(self.name, self.vars, cluster, gcfg)
        return dataclasses.replace(
            self,
            image=dataclasses.replace(self.image, sif_remote=r.expand(self.image.sif_remote)),
            env={k: r.expand(v) for k, v in self.env.items()},
        )

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

        prov_raw = raw.get("provenance", {}) or {}
        provenance = ProvenanceConfig(
            git=str(prov_raw.get("git", "auto")),
            remote=str(prov_raw.get("remote", "origin")),
            branch_prefix=str(prov_raw.get("branch_prefix", "autopush/")),
        )
        if provenance.git not in ("auto", "off"):
            raise ConfigError(f"provenance.git must be auto|off, got '{provenance.git}'")

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
            vars=_check_vars(raw.get("vars"), f"{root / PROJECT_CONFIG_NAME} vars"),
            provenance=provenance,
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
