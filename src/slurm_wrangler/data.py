"""Datasets: pushed ONCE to <storage_root>/projects/<name>/data/<dataset> and
bind-mounted read-only into every job — never part of the per-job code sync."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import ClusterConfig, ConfigError, GlobalConfig, ProjectConfig
from .remote import info, rsync, ssh_mkdirs, ssh_run


def push(project: ProjectConfig, gcfg: GlobalConfig, local_dir: str,
         name: Optional[str] = None, cluster_name: Optional[str] = None,
         delete: bool = False) -> None:
    src = Path(local_dir).expanduser().resolve()
    if not src.is_dir():
        raise ConfigError(f"not a directory: {src}")
    dname = name or src.name
    cluster = gcfg.cluster(cluster_name or (project.defaults or {}).get("cluster"))
    dest = f"{cluster.project_dir(project.name)}/data/{dname}"
    ssh_mkdirs(cluster.login, [dest])
    info(f"pushing dataset '{dname}' -> {cluster.login}:{dest} (resumable)")
    args = ["-a", "--partial", "--info=progress2"]
    if delete:
        args.append("--delete")
    rsync(str(src) + "/", f"{cluster.login}:{dest}/", args)
    declared = {d.name for d in project.data}
    if dname not in declared:
        info(f"dataset pushed. Declare it in sw.yaml so jobs mount it:")
        print(f"\ndata:\n  - name: {dname}\n    src: {src}\n    mount: /data/{dname}\n")
    else:
        mount = next(d.mount for d in project.data if d.name == dname)
        info(f"dataset updated; jobs mount it read-only at {mount}")


def list_data(project: ProjectConfig, gcfg: GlobalConfig,
              cluster_name: Optional[str] = None) -> None:
    cluster = gcfg.cluster(cluster_name or (project.defaults or {}).get("cluster"))
    droot = f"{cluster.project_dir(project.name)}/data"
    out = ssh_run(cluster.login, f"du -sh {droot}/* 2>/dev/null || true").stdout.strip()
    declared = {d.name: d.mount for d in project.data}
    if not out:
        info(f"no datasets on {cluster.name} under {droot}")
    else:
        info(f"datasets on {cluster.name}:")
        for line in out.splitlines():
            size, path = line.split(None, 1)
            name = Path(path).name
            mount = declared.get(name, "(NOT declared in sw.yaml — jobs will not mount it)")
            print(f"  {size:>8}  {name:<24} -> {mount}")
