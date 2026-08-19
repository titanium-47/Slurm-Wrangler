"""Cache presets: named bundles of cache mounts that projects can pull in with
`cache: [{preset: isaac-sim}]` instead of hand-writing every bind.

A cache entry means: "this container path is expensive to repopulate — keep a
persistent copy on the cluster, stage it to node-local scratch before the job,
bind it into the container, and (optionally) sync changes back afterward."
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


def _slug(container_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", container_path).strip("-").lower() or "cache"


# Writeback notes:
#  - "merge" runs plain rsync -a back to the persistent copy: last-writer-wins if
#    two jobs share the cache. Use for caches where that is acceptable.
#  - "ignore-existing" only adds files that don't exist yet: race-free for
#    content-addressed caches (shader blobs, compiled kernels, downloaded models).
PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "huggingface": [
        {"container": "/root/.cache/huggingface", "writeback": True, "writeback_mode": "ignore-existing"},
    ],
    "torch": [
        {"container": "/root/.cache/torch", "writeback": True, "writeback_mode": "ignore-existing"},
    ],
    "pip": [
        {"container": "/root/.cache/pip", "writeback": True, "writeback_mode": "ignore-existing"},
    ],
    # Port of UWLab's run_singularity.sh Isaac Sim cache staging. One shared host
    # tree ("isaac") feeds several container mount points; ensure_dirs pre-creates
    # the subdirectories Isaac Sim crashes without.
    "isaac-sim": [
        {"container": "/isaac-sim/kit/cache", "host_slug": "isaac/kit", "writeback": True,
         "ensure_dirs": ["DerivedDataCache", "nv_shadercache"]},
        {"container": "/root/.cache", "host_slug": "isaac/cache", "writeback": True},
        {"container": "/root/.nv/ComputeCache", "host_slug": "isaac/computecache", "writeback": True},
        {"container": "/root/.nvidia-omniverse/logs", "host_slug": "isaac/logs", "writeback": True},
        {"container": "/root/.local/share/ov/data", "host_slug": "isaac/data", "writeback": True,
         "ensure_dirs": [
             "Kit/Isaac-Sim/5.1/pip3-envs/default",
             "Kit/Isaac-Sim/5.1/exts",
             "documents/Kit/apps/Isaac-Sim/scripts",
             "documents/Kit/shared",
         ]},
        {"container": "/isaac-sim/kit/data", "host_slug": "isaac/data", "writeback": False},
        {"container": "/root/Documents", "host_slug": "isaac/documents", "writeback": True,
         "ensure_dirs": ["Kit/shared"]},
    ],
}


def expand_cache_entries(raw_entries: List[Any]) -> List["CacheConfig"]:
    from .config import CacheConfig, ConfigError  # deferred: avoid import cycle

    out: List[CacheConfig] = []
    for entry in raw_entries:
        if isinstance(entry, str):
            entry = {"preset": entry} if entry in PRESETS else {"container": entry}
        if "preset" in entry:
            name = entry["preset"]
            if name not in PRESETS:
                raise ConfigError(f"Unknown cache preset '{name}'. Known: {', '.join(sorted(PRESETS))}")
            for sub in PRESETS[name]:
                out.append(_mk(CacheConfig, sub))
        else:
            out.append(_mk(CacheConfig, entry))
    return out


def _mk(CacheConfig, d: Dict[str, Any]):
    container = d.get("container", "")
    if not container or not container.startswith("/"):
        from .config import ConfigError
        raise ConfigError(f"cache entry needs an absolute 'container' path: {d}")
    return CacheConfig(
        container=container,
        host_slug=d.get("host_slug") or _slug(container),
        writeback=bool(d.get("writeback", False)),
        writeback_mode=d.get("writeback_mode", "merge"),
        ensure_dirs=list(d.get("ensure_dirs", []) or []),
    )
