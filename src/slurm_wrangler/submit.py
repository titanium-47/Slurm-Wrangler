"""Job submission: resolve parameters (CLI > sw.yaml defaults > cluster config),
ensure image + code snapshot + runtime scripts on the cluster, then submit via
the login node. All parameters cross ssh as SW_* env assignments (shell-quoted),
and the user command as base64 — no quoting hazards across the four-hop chain
(local -> ssh -> sbatch -> srun -> container).
"""
from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional

from .config import ClusterConfig, ConfigError, GlobalConfig, ProjectConfig
from .image import ensure_image
from .remote import RemoteError, info, rsync, ssh_mkdirs, ssh_run, ssh_test, warn
from .sync import apply_big_gate, compute_code_hash, enumerate_files, push_snapshot

JOBS_LOG = "jobs.jsonl"


@dataclass
class SubmitParams:
    cluster: Optional[str] = None
    gpus: Optional[int] = None
    nodes: Optional[int] = None
    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    time: Optional[str] = None
    constraint: Optional[str] = None
    cpus_per_gpu: Optional[int] = None
    mem_per_gpu: Optional[int] = None      # GB
    mem: Optional[str] = None              # total per node, e.g. "240G" (overrides modes)
    name: Optional[str] = None
    requeue: bool = True
    env: Dict[str, str] = field(default_factory=dict)
    exclude_nodes: str = ""
    skip_bad_nodes: bool = False
    race: bool = False


RACE_KEYS = {"partition": "SW_PARTITION", "account": "SW_ACCOUNT",
             "qos": "SW_QOS", "constraint": "SW_CONSTRAINT"}


def race_envs(sw_env: Dict[str, str], targets: List[Dict], group: str) -> List[Dict[str, str]]:
    """One SW_* env per race target (same job name, shared SW_RACE_GROUP)."""
    out = []
    for t in targets:
        bad = set(t) - set(RACE_KEYS)
        if bad:
            raise ConfigError(f"defaults.targets entry has unknown keys {sorted(bad)}; "
                              f"allowed: {sorted(RACE_KEYS)}")
        env = dict(sw_env, SW_RACE_GROUP=group)
        for k, v in t.items():
            env[RACE_KEYS[k]] = str(v or "")
        out.append(env)
    return out


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _resolve(cli_val, project_defaults: Dict, key: str, cluster_slurm: Dict, fallback=None):
    if cli_val is not None:
        return cli_val
    if key in project_defaults and project_defaults[key] is not None:
        return project_defaults[key]
    if key in cluster_slurm and cluster_slurm[key] not in (None, ""):
        return cluster_slurm[key]
    return fallback


def collect_forward_env(gcfg: GlobalConfig, project: ProjectConfig,
                        extra: Dict[str, str]) -> Dict[str, str]:
    """Env vars forwarded into the container: glob patterns from config matched
    against the local environment, plus sw.yaml `env:` statics, plus --env."""
    patterns = list(dict.fromkeys(gcfg.forward_env + project.forward_env))
    out: Dict[str, str] = {}
    for var, val in os.environ.items():
        if any(fnmatch.fnmatchcase(var, pat) for pat in patterns):
            out[var] = val
    out.update(project.env)
    out.update(extra)
    return out


def runtime_hash() -> str:
    h = hashlib.sha256()
    rt = resources.files("slurm_wrangler") / "runtime"
    for name in sorted(p.name for p in rt.iterdir()):
        h.update(name.encode())
        h.update((rt / name).read_bytes())
    return h.hexdigest()[:8]


def ship_runtime(cluster: ClusterConfig, *, dry_run: bool = False) -> str:
    """Ship the runtime scripts, content-addressed, so in-flight jobs keep the
    exact runtime version they were submitted with even if the tool updates."""
    rt_hash = runtime_hash()
    remote_dir = cluster.runtime_dir(rt_hash)
    if dry_run:
        return remote_dir
    if ssh_test(cluster.login, f"-f {remote_dir}/.sw_complete"):
        return remote_dir
    ssh_mkdirs(cluster.login, [remote_dir])
    rt = resources.files("slurm_wrangler") / "runtime"
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for p in rt.iterdir():
            (Path(td) / p.name).write_bytes(p.read_bytes())
        rsync(td + "/", f"{cluster.login}:{remote_dir}/", ["-rt"], quiet=True)
    ssh_run(cluster.login, f"touch {remote_dir}/.sw_complete")
    return remote_dir


def build_static_binds(project: ProjectConfig, cluster: ClusterConfig) -> List[str]:
    """persist dirs, data mounts, cluster FS passthrough. Record format host::container::mode."""
    pdir = cluster.project_dir(project.name)
    binds: List[str] = []
    for d in project.persist:
        d = d.strip("/")
        binds.append(f"{pdir}/persist/{d}::{project.workdir}/{d}::rw")
    for dc in project.data:
        binds.append(f"{pdir}/data/{dc.name}::{dc.mount}::ro")
    binds.append(f"{cluster.mount_dir}::{cluster.mount_dir}::rw")
    return binds


def encode_caches(project: ProjectConfig) -> str:
    recs = []
    for c in project.cache:
        recs.append("|".join([
            c.host_slug, c.container, "1" if c.writeback else "0",
            c.writeback_mode, ",".join(c.ensure_dirs),
        ]))
    return _b64("\n".join(recs)) if recs else ""


def compute_mem_directive(cluster: ClusterConfig, params: SubmitParams, gpus: int,
                          slurm: Dict) -> str:
    if params.mem:
        return f"#SBATCH --mem={params.mem}"
    mode = slurm.get("mem_mode", "per_node_from_gpu")
    if mode == "per_cpu":
        per_cpu = int(slurm.get("mem_per_cpu_gb", 25))
        return f"#SBATCH --mem-per-cpu={per_cpu}G"
    if gpus <= 0:
        return f"#SBATCH --mem={int(slurm.get('mem_cpu_job_gb', 32))}G"
    per_gpu = params.mem_per_gpu if params.mem_per_gpu is not None else int(slurm.get("mem_per_gpu_gb", 60))
    return f"#SBATCH --mem={gpus * per_gpu}G"


def submit(project: ProjectConfig, gcfg: GlobalConfig, params: SubmitParams,
           command: List[str], *, dry_run: bool = False, allow_large: bool = False,
           no_build: bool = False) -> Optional[str]:
    if not command:
        raise ConfigError("no command given — usage: sw run [flags] -- <command...>")
    pdefaults = project.defaults or {}
    cluster = gcfg.cluster(params.cluster or pdefaults.get("cluster"))
    project = project.resolved(cluster, gcfg)  # expand sw.yaml {placeholders}
    slurm = cluster.slurm or {}

    gpus = int(_resolve(params.gpus, pdefaults, "gpus", {}, 1))
    nodes = int(_resolve(params.nodes, pdefaults, "nodes", {}, 1))
    partition = _resolve(params.partition, pdefaults, "partition", slurm, "") or ""
    # empty partition = use the cluster's default partition (Tillicum convention)
    account = _resolve(params.account, pdefaults, "account", slurm, "")
    qos = _resolve(params.qos, pdefaults, "qos", slurm, "")
    time_lim = _resolve(params.time, pdefaults, "time", slurm, "24:00:00")
    cpus_per_gpu = int(_resolve(params.cpus_per_gpu, pdefaults, "cpus_per_gpu", slurm, 6))
    cpus_per_task = max(1, gpus) * cpus_per_gpu

    # Constraint precedence: CLI > project default > forced cluster constraint > slurm default
    if params.constraint is not None:
        constraint = params.constraint
    elif pdefaults.get("constraint") is not None:
        constraint = pdefaults["constraint"]
    elif cluster.force_constraint and gpus > 0:
        constraint = cluster.force_constraint
    else:
        constraint = slurm.get("constraint", "")
    if gpus == 0:
        constraint = params.constraint or ""

    if nodes > 1 and project.launcher != "torchrun":
        warn("multi-node job with launcher=plain: every node runs the same command "
             "independently. Use launcher: torchrun for distributed training.")

    # --- ensure remote pieces ---
    sif_path = ensure_image(project, cluster, auto_build=not no_build, dry_run=dry_run)
    entries = enumerate_files(project)
    entries = apply_big_gate(project, entries, allow_all=allow_large)
    code_hash = compute_code_hash(project, entries)
    pdir = cluster.project_dir(project.name)
    if not dry_run:
        ssh_mkdirs(cluster.login, [pdir, f"{pdir}/persist", cluster.slurm_logs_dir(project.name)]
                   + [f"{pdir}/persist/{d.strip('/')}" for d in project.persist])
    code_dir = push_snapshot(project, cluster, entries, code_hash, dry_run=dry_run)
    runtime_dir = ship_runtime(cluster, dry_run=dry_run)

    # --- assemble SW_* env ---
    cmd_str = " ".join(shlex.quote(c) if re.search(r"\s", c) else c for c in command)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = params.name or f"sw-{project.name}-{stamp}"
    fwd_env = collect_forward_env(gcfg, project, params.env)
    binds = build_static_binds(project, cluster)

    sw_env: Dict[str, str] = {
        "SW_PROJECT": project.name,
        "SW_PROJECT_DIR": pdir,
        "SW_CODE_DIR": code_dir,
        "SW_SIF": sif_path,
        "SW_RUNTIME_DIR": runtime_dir,
        "SW_JOB_NAME": job_name,
        "SW_NODES": str(nodes),
        "SW_GPUS_PER_NODE": str(gpus),
        "SW_PARTITION": str(partition or ""),
        "SW_ACCOUNT": str(account or ""),
        "SW_QOS": str(qos or ""),
        "SW_TIME": str(time_lim),
        "SW_CONSTRAINT": str(constraint or ""),
        "SW_CPUS_PER_TASK": str(cpus_per_task),
        "SW_MEM_DIRECTIVE": compute_mem_directive(cluster, params, gpus, slurm),
        "SW_SLURM_LOGS_DIR": cluster.slurm_logs_dir(project.name),
        "SW_SIGNAL_SECS": str(cluster.requeue_signal_secs),
        "SW_REQUEUE": "1" if params.requeue else "0",
        "SW_LAUNCHER": project.launcher,
        "SW_WORKDIR": project.workdir,
        "SW_IMAGE_KIND": project.image.kind,
        "SW_CMD_B64": _b64(cmd_str),
        "SW_BINDS_B64": _b64("\n".join(binds)),
        "SW_CACHES_B64": encode_caches(project),
        "SW_ENV_FWD_B64": _b64("\n".join(f"{k}={v}" for k, v in fwd_env.items())),
        "SW_PRESTAGE": project.prestage,
        "SW_EXCLUDE_NODES": params.exclude_nodes,
        "SW_SKIP_BAD_NODES": "1" if params.skip_bad_nodes else "0",
    }

    if "wandb" in cmd_str and "WANDB_API_KEY" not in fwd_env:
        warn("command mentions wandb but WANDB_API_KEY is not in the forwarded env "
             "(export it locally so it reaches the cluster)")

    targets = pdefaults.get("targets") or []
    if params.race and not targets:
        raise ConfigError("--race needs a non-empty `defaults.targets` list in sw.yaml")
    envs = race_envs(sw_env, targets, job_name) if params.race else [sw_env]

    def _cmd(env: Dict[str, str]) -> str:
        assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        return f"{assignments} bash {shlex.quote(runtime_dir + '/submit_slurm.sh')}"

    if dry_run:
        info("[dry-run] resolved submission:")
        for i, env in enumerate(envs):
            if params.race:
                info(f"  --- race target {i + 1}/{len(envs)} ---")
            for k in ("SW_JOB_NAME", "SW_PARTITION", "SW_ACCOUNT", "SW_QOS", "SW_CONSTRAINT",
                      "SW_NODES", "SW_GPUS_PER_NODE", "SW_CPUS_PER_TASK", "SW_MEM_DIRECTIVE",
                      "SW_TIME", "SW_CODE_DIR", "SW_SIF", "SW_WORKDIR", "SW_LAUNCHER"):
                info(f"  {k} = {env[k]}")
        info(f"  command = {cmd_str}")
        info(f"  forwarded env: {', '.join(fwd_env) or '(none)'}")
        info(f"  binds:\n" + "\n".join(f"    {b.replace('::', ' -> ', 1).replace('::', ' ')}" for b in binds))
        return None

    first_id = None
    for env in envs:
        proc = ssh_run(cluster.login, _cmd(env), timeout=900)
        out = proc.stdout.strip()
        print(out)
        m = re.search(r"Submitted batch job (\d+)", out)
        job_id = m.group(1) if m else None
        first_id = first_id or job_id
        rec = {
            "job_id": job_id, "cluster": cluster.name, "name": job_name, "cmd": cmd_str,
            "code_hash": code_hash, "sif": sif_path, "partition": env["SW_PARTITION"],
            "gpus": gpus, "nodes": nodes, "time": time_lim,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "slurm_logs_dir": cluster.slurm_logs_dir(project.name),
        }
        if params.race:
            rec["group"] = job_name
        _record_job(project, rec)
        if job_id:
            log = f"{cluster.slurm_logs_dir(project.name)}/{job_name}-{job_id}.out"
            info(f"submitted job {job_id} on {cluster.name}"
                 + (f" (partition {env['SW_PARTITION'] or '<default>'})" if params.race else ""))
            info(f"  follow logs:  sw logs {job_id}")
            info(f"  log file:     {cluster.login}:{log}")
    info(f"  queue status: sw status")
    return first_id


def _record_job(project: ProjectConfig, record: Dict) -> None:
    path = project.state_dir / JOBS_LOG
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_job_records(project: ProjectConfig, n: int = 20) -> List[Dict]:
    path = project.state_dir / JOBS_LOG
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
