"""sw — slurm_wrangler CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, GlobalConfig, ProjectConfig, find_project_root
from .remote import RemoteError, die, info
from .submit import SubmitParams, submit
from .sync import SyncGateError


def _parse_env(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ConfigError(f"--env expects KEY=VALUE, got '{p}'")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def _project_or_none():
    try:
        return ProjectConfig.load()
    except ConfigError:
        return None


def _add_submit_flags(p):
    p.add_argument("--cluster", help="cluster name (default: sw.yaml defaults, then global)")
    p.add_argument("--gpus", type=int, help="GPUs per node (0 = CPU job)")
    p.add_argument("--nodes", type=int, help="number of nodes")
    p.add_argument("--partition", help="SLURM partition")
    p.add_argument("--account", help="SLURM account")
    p.add_argument("--qos", help="SLURM QOS (e.g. ckpt-gpu)")
    p.add_argument("--time", help="wall-clock limit HH:MM:SS")
    p.add_argument("--constraint", help="GPU feature expression, e.g. 'a40|a100'")
    p.add_argument("--cpus-per-gpu", type=int, help="CPUs per GPU")
    p.add_argument("--mem-per-gpu", type=int, help="GB of RAM per GPU")
    p.add_argument("--mem", help="total memory per node (e.g. 240G); overrides per-gpu")
    p.add_argument("--name", help="job name (default sw-<project>-<timestamp>)")
    p.add_argument("--no-requeue", action="store_true", help="disable auto-requeue machinery")
    p.add_argument("--env", action="append", metavar="K=V", help="extra env var for the job (repeatable)")
    p.add_argument("--exclude-nodes", default="", help="comma list of extra nodes to exclude")
    p.add_argument("--skip-bad-nodes", action="store_true", help="ignore the bad-node registry")
    p.add_argument("--allow-large", action="store_true", help="approve all >1GB items this run")
    p.add_argument("--no-build", action="store_true", help="fail instead of auto-building a missing image")
    p.add_argument("--dry-run", action="store_true", help="resolve and print everything, submit nothing")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="sw",
        description="slurm_wrangler: run any local project on Hyak/Tillicum SLURM. "
                    "One-time: `sw setup` then per-project `sw init`.",
    )
    ap.add_argument("--version", action="version", version=f"slurm_wrangler {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("setup", help="interactive global setup (NetID, clusters, ssh config)")
    p_init = sub.add_parser("init", help="interactive project onboarding (writes sw.yaml)")
    p_init.add_argument("path", nargs="?", default=".", help="project root (default: cwd)")

    p_doc = sub.add_parser("doctor", help="preflight checks (local toolchain, ssh, remote dirs)")
    p_doc.add_argument("--cluster")

    p_run = sub.add_parser("run", help="submit a job: sw run [flags] -- <command...>")
    _add_submit_flags(p_run)
    p_run.add_argument("command", nargs=argparse.REMAINDER,
                       help="command to run in the container (prefix with --)")

    p_env = sub.add_parser("env", help="image lifecycle")
    env_sub = p_env.add_subparsers(dest="env_cmd", required=True)
    env_sub.add_parser("status", help="show env hash and local/remote image state").add_argument("--cluster")
    env_sub.add_parser("build", help="build docker image + convert to SIF locally")
    env_sub.add_parser("push", help="build if needed and upload the SIF").add_argument("--cluster")

    p_sync = sub.add_parser("sync", help="sync the code snapshot (also happens on every run)")
    p_sync.add_argument("--cluster")
    p_sync.add_argument("--review", action="store_true", help="re-ask all large-directory decisions")
    p_sync.add_argument("--allow-large", action="store_true")
    p_sync.add_argument("--dry-run", action="store_true")

    p_data = sub.add_parser("data", help="datasets (pushed once, mounted read-only)")
    data_sub = p_data.add_subparsers(dest="data_cmd", required=True)
    dp = data_sub.add_parser("push", help="rsync a directory to the cluster data area")
    dp.add_argument("dir")
    dp.add_argument("--name", help="dataset name (default: dir basename)")
    dp.add_argument("--cluster")
    dp.add_argument("--delete", action="store_true", help="mirror-delete remote extras")
    dl = data_sub.add_parser("list", help="list remote datasets + mount status")
    dl.add_argument("--cluster")

    p_st = sub.add_parser("status", help="squeue + recent submissions from this project")
    p_st.add_argument("--cluster")
    p_st.add_argument("-n", type=int, default=15, help="recent submissions to show")

    p_logs = sub.add_parser("logs", help="show/tail a job's SLURM log")
    p_logs.add_argument("job_id")
    p_logs.add_argument("-f", "--follow", action="store_true")
    p_logs.add_argument("--err", action="store_true", help="show stderr stream instead")
    p_logs.add_argument("--cluster")

    p_cancel = sub.add_parser("cancel", help="scancel a job (disables its auto-requeue first)")
    p_cancel.add_argument("job_id")
    p_cancel.add_argument("--cluster")

    p_prune = sub.add_parser("prune", help="delete old code snapshots on the cluster")
    p_prune.add_argument("--keep", type=int, default=5)
    p_prune.add_argument("--cluster")

    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 1

    try:
        return _dispatch(args)
    except (ConfigError, SyncGateError, RemoteError) as e:
        die(str(e))
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    return 0


def _dispatch(args) -> int:
    if args.cmd == "setup":
        from .wizard import run_setup
        run_setup()
        return 0

    if args.cmd == "init":
        from .wizard import run_init
        run_init(Path(args.path))
        return 0

    gcfg = GlobalConfig.load()

    if args.cmd == "doctor":
        from .jobs import doctor
        return 1 if doctor(_project_or_none(), gcfg, args.cluster) else 0

    if args.cmd == "status":
        from .jobs import status
        status(_project_or_none(), gcfg, args.cluster, args.n)
        return 0

    if args.cmd == "logs":
        from .jobs import logs
        logs(_project_or_none(), gcfg, args.job_id, follow=args.follow,
             cluster_name=args.cluster, stream="err" if args.err else "out")
        return 0

    if args.cmd == "cancel":
        from .jobs import cancel
        cancel(_project_or_none(), gcfg, args.job_id, args.cluster)
        return 0

    # Everything below needs a project.
    project = ProjectConfig.load()

    if args.cmd == "run":
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        params = SubmitParams(
            cluster=args.cluster, gpus=args.gpus, nodes=args.nodes,
            partition=args.partition, account=args.account, qos=args.qos,
            time=args.time, constraint=args.constraint,
            cpus_per_gpu=args.cpus_per_gpu, mem_per_gpu=args.mem_per_gpu,
            mem=args.mem, name=args.name, requeue=not args.no_requeue,
            env=_parse_env(args.env), exclude_nodes=args.exclude_nodes,
            skip_bad_nodes=args.skip_bad_nodes,
        )
        submit(project, gcfg, params, command, dry_run=args.dry_run,
               allow_large=args.allow_large, no_build=args.no_build)
        return 0

    if args.cmd == "env":
        from .image import (build_docker, build_sif, docker_tag, env_hash,
                            push_image, remote_has_image, remote_sif_path)
        ehash = env_hash(project)
        if args.env_cmd == "status":
            info(f"env hash: {ehash}  (image {docker_tag(project, ehash)})")
            import subprocess
            have_local = subprocess.run(
                ["docker", "image", "inspect", docker_tag(project, ehash)],
                capture_output=True).returncode == 0
            info(f"local docker image: {'present' if have_local else 'not built'}")
            sif = project.state_dir / "exports" / f"env-{ehash}.sif"
            info(f"local SIF: {sif if sif.exists() else 'not converted'}")
            cluster = gcfg.cluster(args.cluster or (project.defaults or {}).get("cluster"))
            project = project.resolved(cluster, gcfg)
            present = remote_has_image(project, cluster, ehash) if project.image.kind != "prebuilt" else True
            info(f"on {cluster.name}: {'present' if present else 'MISSING'} "
                 f"({remote_sif_path(project, cluster, ehash)})")
        elif args.env_cmd == "build":
            build_docker(project, ehash)
            build_sif(project, ehash, project.state_dir / "exports")
        elif args.env_cmd == "push":
            cluster = gcfg.cluster(args.cluster or (project.defaults or {}).get("cluster"))
            push_image(project.resolved(cluster, gcfg), cluster, ehash)
        return 0

    if args.cmd == "sync":
        from .sync import (apply_big_gate, compute_code_hash, enumerate_files,
                           push_snapshot)
        cluster = gcfg.cluster(args.cluster or (project.defaults or {}).get("cluster"))
        entries = enumerate_files(project)
        entries = apply_big_gate(project, entries, allow_all=args.allow_large,
                                 reask=args.review)
        code_hash = compute_code_hash(project, entries)
        info(f"code hash: {code_hash}")
        push_snapshot(project, cluster, entries, code_hash, dry_run=args.dry_run)
        return 0

    if args.cmd == "data":
        from . import data as data_mod
        if args.data_cmd == "push":
            data_mod.push(project, gcfg, args.dir, name=args.name,
                          cluster_name=args.cluster, delete=args.delete)
        else:
            data_mod.list_data(project, gcfg, args.cluster)
        return 0

    if args.cmd == "prune":
        from .sync import prune_snapshots
        cluster = gcfg.cluster(args.cluster or (project.defaults or {}).get("cluster"))
        prune_snapshots(project, cluster, keep=args.keep)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
