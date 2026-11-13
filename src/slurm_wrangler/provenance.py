"""Run provenance: pin the exact code a job runs to a pushed git commit, and
collect the metadata (`sw_meta.json`) that lets a run be reproduced from W&B.

`sw run` syncs the *working tree* (tracked + untracked-not-ignored files), so a
job often runs uncommitted code. `git_provenance` builds a commit of exactly the
synced files — HEAD plus local edits — in a temporary index (the user's branch,
index and working tree are never touched), and pushes it to
`autopush/<timestamp>` when it isn't already on the remote.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .remote import info, warn

AUTOPUSH_CACHE = "autopush.json"
MAX_PUSH_FILE_BYTES = 95 * 1024 * 1024   # GitHub hard-rejects files > 100 MB
SECRET_PATTERNS = ("*KEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*", "*PASSWD*", "*CREDENTIAL*")


class GitError(Exception):
    pass


def _git(root: Path, *args: str, env: Optional[Dict[str, str]] = None,
         input: Optional[bytes] = None, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          input=input, env=env)
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout.decode(errors="replace").strip()


def redact_env(env: Dict[str, str]) -> Dict[str, str]:
    """Mask values of secret-looking keys (WANDB_API_KEY, HF_TOKEN, ...)."""
    return {k: ("<redacted>" if any(fnmatch.fnmatchcase(k.upper(), p) for p in SECRET_PATTERNS)
                else v) for k, v in env.items()}


def browse_url(remote_url: str, commit: str) -> str:
    """git@github.com:o/r.git -> https://github.com/o/r/tree/<commit> (best effort)."""
    m = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+?)(?:\.git)?/?$", remote_url)
    if m:
        base = f"https://{m.group(1)}/{m.group(2)}"
    elif remote_url.startswith(("https://", "http://")):
        base = re.sub(r"(\.git)?/?$", "", remote_url)
    else:
        return ""
    return f"{base}/tree/{commit}"


def _pick_remote(root: Path, configured: str) -> str:
    branch = _git(root, "symbolic-ref", "-q", "--short", "HEAD", check=False)
    if branch:
        up = _git(root, "config", f"branch.{branch}.remote", check=False)
        if up and up != ".":
            return up
    return configured


def _synced_tree(root: Path, prefix: str, relpaths: Sequence[str]) -> str:
    """Tree object = HEAD with every synced file replaced by its on-disk content
    and tracked-but-deleted files removed. Uses a throwaway index."""
    real_index = _git(root, "rev-parse", "--path-format=absolute", "--git-path", "index")
    with tempfile.TemporaryDirectory(prefix="sw_git_") as td:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(td, "index"))
        # Seed from the real index so its stat cache spares re-hashing unchanged
        # files; read-tree -m then resets the content to HEAD. copy2 keeps the
        # index mtime, which git's racy-clean detection depends on.
        if os.path.exists(real_index):
            shutil.copy2(real_index, env["GIT_INDEX_FILE"])
        if subprocess.run(["git", "-C", str(root), "read-tree", "-m", "HEAD"], env=env,
                          capture_output=True).returncode != 0:
            _git(root, "read-tree", "HEAD", env=env)
        if relpaths:
            spec = b"\0".join((prefix + p).encode() for p in relpaths) + b"\0"
            _git(root, "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul",
                 "--", env=env, input=spec)
        # tracked in HEAD but gone from disk (only inside the project subtree)
        deleted = [p for p in _git(root, "ls-files", "-z", "--deleted", "--", prefix or ".",
                                   env=env).split("\0") if p]
        if deleted:
            _git(root, "rm", "-q", "--cached", "--pathspec-from-file=-", "--pathspec-file-nul",
                 env=env, input=b"\0".join(p.encode() for p in deleted) + b"\0")
        return _git(root, "write-tree", env=env)


def _oversized(root: Path, base: str, tree: str) -> List[str]:
    out = _git(root, "diff-tree", "-r", "-z", "--no-renames", "--diff-filter=AM",
               "--name-only", base, tree)
    big = []
    for p in filter(None, out.split("\0")):
        f = root / p
        try:
            if f.is_file() and not f.is_symlink() and f.stat().st_size > MAX_PUSH_FILE_BYTES:
                big.append(p)
        except OSError:
            pass
    return big


def git_provenance(project_root: Path, state_dir: Path, relpaths: Sequence[str], *,
                   job_name: str = "", remote: str = "origin",
                   branch_prefix: str = "autopush/", push: bool = True,
                   dry_run: bool = False) -> Dict:
    """Return the git provenance dict for the synced files (empty if not a git repo).

    Keys: commit, base_commit, branch, dirty, remote, remote_url, autopush_branch,
    pushed, commit_url, note.
    """
    root = Path(project_root).resolve()
    try:
        top = Path(_git(root, "rev-parse", "--show-toplevel"))
        head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    except (GitError, FileNotFoundError):
        return {}
    prefix = os.path.relpath(root, top)
    prefix = "" if prefix == "." else prefix.rstrip("/") + "/"

    remote = _pick_remote(top, remote)
    remote_url = _git(top, "remote", "get-url", remote, check=False)
    branch = _git(top, "symbolic-ref", "-q", "--short", "HEAD", check=False) or None
    tree = _synced_tree(top, prefix, relpaths)
    dirty = tree != _git(top, "rev-parse", "HEAD^{tree}")
    rec: Dict = {"commit": head, "base_commit": head, "branch": branch, "dirty": dirty,
                 "remote": remote, "remote_url": remote_url, "autopush_branch": None,
                 "pushed": False, "commit_url": "", "note": ""}

    head_on_remote = bool(_git(top, "for-each-ref", "--contains", head,
                               "--format=%(refname)", f"refs/remotes/{remote}", check=False))
    if not dirty and head_on_remote:
        rec["pushed"] = True
        rec["commit_url"] = browse_url(remote_url, head)
        return rec

    cache_path = Path(state_dir) / AUTOPUSH_CACHE
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    key = f"{head}:{tree}:{remote}"
    hit = cache.get(key)
    if hit and _git(top, "cat-file", "-t", hit["commit"], check=False) == "commit":
        rec.update(commit=hit["commit"], autopush_branch=hit["branch"], pushed=True,
                   commit_url=browse_url(remote_url, hit["commit"]))
        info(f"code pinned to {hit['commit'][:10]} (already pushed: {remote}/{hit['branch']})")
        return rec

    if dirty:
        msg = f"sw autopush: {job_name or 'sw run'}\n\nWorking tree as synced by slurm_wrangler, on top of {head}."
        rec["commit"] = _git(top, "commit-tree", tree, "-p", head, "-m", msg)
    rec["commit_url"] = browse_url(remote_url, rec["commit"])
    autobranch = f"{branch_prefix}{time.strftime('%Y%m%d-%H%M%S')}"

    if not remote_url:
        rec["note"] = f"no git remote '{remote}'; commit exists only locally"
        warn(f"provenance: {rec['note']} ({rec['commit'][:10]})")
        return rec
    big = _oversized(top, head, tree) if dirty else []
    if big:
        rec["note"] = f"not pushed: files over {MAX_PUSH_FILE_BYTES >> 20} MB: {big[:5]}"
        warn(f"provenance: {rec['note']} — commit {rec['commit'][:10]} exists only locally")
        return rec
    if not push or dry_run:
        rec["autopush_branch"] = autobranch
        rec["note"] = "dry run: not pushed" if dry_run else "--no-autopush: not pushed"
        info(f"code would be pinned to {rec['commit'][:10]} -> {remote}/{autobranch} ({rec['note']})")
        return rec

    proc = subprocess.run(["git", "-C", str(top), "push", "-q", remote,
                           f"{rec['commit']}:refs/heads/{autobranch}"],
                          capture_output=True, timeout=300)
    if proc.returncode != 0:
        rec["note"] = "push failed: " + proc.stderr.decode(errors="replace").strip()[-300:]
        warn(f"provenance: {rec['note']}\n  run will record local-only commit {rec['commit'][:10]}")
        return rec
    rec.update(autopush_branch=autobranch, pushed=True)
    cache[key] = {"commit": rec["commit"], "branch": autobranch}
    try:
        cache_path.write_text(json.dumps(cache, indent=1))
    except OSError:
        pass
    info(f"code pinned to {rec['commit'][:10]} -> pushed {remote}/{autobranch}")
    return rec


def sw_self_commit() -> str:
    """slurm_wrangler's own commit when installed from a git checkout (editable)."""
    pkg = Path(__file__).resolve().parent
    try:
        sha = _git(pkg, "rev-parse", "HEAD")
        dirty = _git(pkg, "status", "--porcelain", "--untracked-files=no", "--", ".")
        return sha + ("-dirty" if dirty else "")
    except (GitError, FileNotFoundError):
        return ""


def repro_command(git: Dict, sw_argv: Sequence[str]) -> str:
    argv = " ".join(shlex.quote(a) for a in sw_argv)
    if not git:
        return f"sw {argv}"
    ref = git.get("autopush_branch") or git.get("branch") or ""
    fetch = f"git fetch {git.get('remote', 'origin')} {ref} && " if ref else ""
    return f"{fetch}git checkout {git['commit']} && sw {argv}"
