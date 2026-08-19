"""Job monitoring: status, logs, cancel, doctor."""
from __future__ import annotations

import shlex
import shutil
import subprocess
from typing import Optional

from .config import ClusterConfig, GlobalConfig, ProjectConfig
from .remote import info, ssh_check_socket, ssh_run, warn
from .submit import load_job_records


def _cluster_for(project: Optional[ProjectConfig], gcfg: GlobalConfig,
                 cluster_name: Optional[str]) -> ClusterConfig:
    default = (project.defaults or {}).get("cluster") if project else None
    return gcfg.cluster(cluster_name or default)


def status(project: Optional[ProjectConfig], gcfg: GlobalConfig,
           cluster_name: Optional[str] = None, n: int = 15) -> None:
    cluster = _cluster_for(project, gcfg, cluster_name)
    user = gcfg.user or "$USER"
    out = ssh_run(cluster.login,
                  f'squeue -u {user} -o "%.10i %.12P %.28j %.4t %.11M %.5D %R"').stdout.rstrip()
    print(f"--- queue on {cluster.name} ---")
    print(out if out.strip() else "(empty)")
    if project:
        recs = load_job_records(project, n)
        if recs:
            print(f"\n--- recent submissions from this project ({project.name}) ---")
            for r in recs:
                print(f"  {r.get('job_id') or '?':>9}  {r.get('cluster', ''):<9} "
                      f"{r.get('submitted_at', '')}  {r.get('cmd', '')[:80]}")


def logs(project: Optional[ProjectConfig], gcfg: GlobalConfig, job_id: str,
         follow: bool = False, cluster_name: Optional[str] = None, stream: str = "out") -> None:
    cluster = _cluster_for(project, gcfg, cluster_name)
    logs_dir = None
    if project:
        for r in reversed(load_job_records(project, 100)):
            if r.get("job_id") == job_id:
                logs_dir = r.get("slurm_logs_dir")
                cluster = gcfg.cluster(r.get("cluster"))
                break
        if logs_dir is None:
            logs_dir = cluster.slurm_logs_dir(project.name)
    if logs_dir is None:
        warn("no project context; searching the whole storage root")
        logs_dir = cluster.storage_root
    pattern = f"{logs_dir}/*-{job_id}.{stream}"
    if follow:
        # tail -F over the control socket: instant, survives requeues (append mode)
        cmd = ["ssh", cluster.login, f"tail -n 100 -F $(ls -1 {pattern} 2>/dev/null | head -1)"]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
    else:
        out = ssh_run(cluster.login,
                      f"f=$(ls -1 {pattern} 2>/dev/null | head -1); "
                      f"if [ -n \"$f\" ]; then echo \"== $f ==\"; tail -n 200 \"$f\"; "
                      f"else echo 'no log file matching {pattern}'; fi").stdout
        print(out)


def cancel(project: Optional[ProjectConfig], gcfg: GlobalConfig, job_id: str,
           cluster_name: Optional[str] = None) -> None:
    cluster = _cluster_for(project, gcfg, cluster_name)
    # Disable requeue first so the classifier honors the operator cancel.
    ssh_run(cluster.login,
            f"scontrol update JobId={shlex.quote(job_id)} Requeue=0 2>/dev/null; "
            f"scancel {shlex.quote(job_id)}")
    info(f"cancelled job {job_id} on {cluster.name} (requeue disabled first)")


def doctor(project: Optional[ProjectConfig], gcfg: GlobalConfig,
           cluster_name: Optional[str] = None) -> int:
    """Preflight checks. Returns number of failures."""
    failures = 0

    def check(label: str, ok: bool, fix: str = "") -> None:
        nonlocal failures
        mark = "\033[32mOK\033[0m" if ok else "\033[31mFAIL\033[0m"
        print(f"  [{mark}] {label}")
        if not ok:
            failures += 1
            if fix:
                print(f"         fix: {fix}")

    print("local toolchain:")
    check("rsync installed", shutil.which("rsync") is not None, "apt install rsync")
    check("docker installed", shutil.which("docker") is not None,
          "needed to build images (not for prebuilt kind)")
    check("apptainer installed", shutil.which("apptainer") is not None,
          "sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt install -y apptainer")

    print("global config:")
    check(f"user set ('{gcfg.user}')", bool(gcfg.user), "run `sw setup`")
    check(f"clusters configured ({', '.join(sorted(gcfg.clusters)) or 'none'})",
          bool(gcfg.clusters), "run `sw setup`")

    clusters = list(gcfg.clusters.values())
    if cluster_name:
        clusters = [gcfg.cluster(cluster_name)]
    for cluster in clusters:
        print(f"cluster {cluster.name} ({cluster.login}):")
        socket = ssh_check_socket(cluster.login)
        if not socket:
            print(f"         (no ControlMaster socket; the next check may trigger Duo)")
        try:
            proc = ssh_run(cluster.login, "echo ok", timeout=30)
            check("ssh reachable", proc.stdout.strip() == "ok",
                  f"ssh {cluster.login} in a real terminal (Duo), check ~/.ssh/config")
        except Exception as e:
            check("ssh reachable", False, f"ssh {cluster.login} manually; {e}")
            continue
        proc = ssh_run(cluster.login, f"mkdir -p {shlex.quote(cluster.storage_root)} && echo ok",
                       check=False)
        check(f"storage root writable ({cluster.storage_root})",
              proc.returncode == 0 and "ok" in proc.stdout)
        proc = ssh_run(cluster.login, "command -v sbatch >/dev/null && echo ok || echo no", check=False)
        check("sbatch available", "ok" in (proc.stdout or ""))
        proc = ssh_run(cluster.login,
                       "command -v singularity >/dev/null || command -v apptainer >/dev/null "
                       "&& echo ok || echo no", check=False)
        check("singularity/apptainer available (login node)", "ok" in (proc.stdout or ""),
              "compute nodes may still have it; check `module load apptainer` needs")

    if project:
        print(f"project {project.name}:")
        if project.image.kind == "conda":
            envf = project.root / project.image.environment
            check(f"env file exists ({project.image.environment})", envf.exists(),
                  f"conda env export -n <env> --no-builds > {project.image.environment}")
        elif project.image.kind == "dockerfile":
            check(f"dockerfile exists ({project.image.dockerfile})",
                  (project.root / project.image.dockerfile).exists())
        from .image import env_hash
        try:
            eh = env_hash(project)
            print(f"  [\033[32mOK\033[0m] env hash: {eh}")
        except Exception as e:
            check("env hash computable", False, str(e))

    print(f"\n{failures} problem(s) found." if failures else "\nAll checks passed.")
    return failures
