"""Code sync: enumerate project files (honoring .gitignore + .dockerignore +
extra excludes), gate suspiciously large top-level directories behind a one-time
user confirmation, content-hash the result, and push a deduplicated snapshot to
the cluster at  <storage_root>/projects/<name>/code/<hash12>/.

Snapshots are content-addressed: the same code state is only ever shipped and
stored once, no matter how many jobs you launch from it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (ClusterConfig, ProjectConfig, is_interactive, load_state,
                     save_state, REASK_GROWTH_FACTOR)
from .remote import info, rsync, ssh_mkdirs, ssh_run, ssh_test, warn

SYNC_STATE_FILE = "sync.local.yaml"
HASH_CACHE_FILE = "hashcache.json"


class SyncGateError(Exception):
    """Raised in non-interactive mode when a large item has no recorded decision."""


@dataclass
class FileEntry:
    relpath: str
    size: int
    is_symlink: bool


# ---------------------------------------------------------------------------
# Ignore-pattern matching (pragmatic subset of gitignore / dockerignore)
# ---------------------------------------------------------------------------
class IgnoreMatcher:
    """Gitignore-style matcher. `rooted=True` gives dockerignore-ish semantics
    (patterns are anchored to the root unless they contain a wildcard prefix)."""

    def __init__(self, patterns: List[str], rooted: bool = False, base: str = ""):
        # base: dir (relative to project root, '' for root) the patterns are scoped to
        self.rules = []  # (negated, dir_only, regex)
        self.base = base.rstrip("/")
        for raw in patterns:
            line = raw.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            line = line.rstrip()
            negated = line.startswith("!")
            if negated:
                line = line[1:]
            if line.startswith("\\"):
                line = line[1:]
            dir_only = line.endswith("/")
            line = line.rstrip("/")
            if not line:
                continue
            anchored = rooted or line.startswith("/") or ("/" in line)
            line = line.lstrip("/")
            regex = self._translate(line, anchored)
            self.rules.append((negated, dir_only, re.compile(regex)))

    @staticmethod
    def _translate(pat: str, anchored: bool) -> str:
        out = []
        i = 0
        while i < len(pat):
            c = pat[i]
            if c == "*":
                if pat[i:i + 2] == "**":
                    out.append(".*")
                    i += 2
                    if i < len(pat) and pat[i] == "/":
                        i += 1
                    continue
                out.append("[^/]*")
            elif c == "?":
                out.append("[^/]")
            elif c == "[":
                j = pat.find("]", i)
                if j == -1:
                    out.append(re.escape(c))
                else:
                    out.append(pat[i:j + 1])
                    i = j
            else:
                out.append(re.escape(c))
            i += 1
        body = "".join(out)
        prefix = "" if anchored else "(?:.*/)?"
        return f"^{prefix}{body}$"

    def _rel_to_base(self, relpath: str) -> Optional[str]:
        if not self.base:
            return relpath
        if relpath == self.base or relpath.startswith(self.base + "/"):
            sub = relpath[len(self.base):].lstrip("/")
            return sub or None
        return None

    def match(self, relpath: str, is_dir: bool) -> Optional[bool]:
        """Return True (ignore), False (explicitly un-ignored), or None (no rule)."""
        sub = self._rel_to_base(relpath)
        if sub is None:
            return None
        verdict: Optional[bool] = None
        for negated, dir_only, regex in self.rules:
            if dir_only and not is_dir:
                # dir-only patterns still match files *inside* a matching dir;
                # handled by the caller pruning directories.
                continue
            if regex.match(sub):
                verdict = not negated
        return verdict


def _load_ignore_file(path: Path, rooted: bool, base: str) -> Optional[IgnoreMatcher]:
    if not path.exists():
        return None
    with open(path, errors="replace") as f:
        return IgnoreMatcher(f.read().splitlines(), rooted=rooted, base=base)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
ALWAYS_EXCLUDE = {".git", ".sw"}


def _git_enumerate(root: Path) -> Optional[List[str]]:
    """tracked + untracked-but-not-ignored files, via git (perfect gitignore semantics)."""
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others",
             "--exclude-standard", "-z"],
            capture_output=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [p.decode("utf-8", "replace") for p in out.split(b"\0") if p]


def _walk_enumerate(root: Path, respect_gitignore: bool) -> List[str]:
    """Fallback for non-git projects: os.walk with per-directory .gitignore files."""
    matchers: List[IgnoreMatcher] = []
    if respect_gitignore:
        m = _load_ignore_file(root / ".gitignore", rooted=False, base="")
        if m:
            matchers.append(m)
    results: List[str] = []

    def ignored(rel: str, is_dir: bool) -> bool:
        verdict = None
        for m in matchers:
            v = m.match(rel, is_dir)
            if v is not None:
                verdict = v
        return bool(verdict)

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "" if rel_dir == "." else rel_dir
        if respect_gitignore:
            sub = _load_ignore_file(Path(dirpath) / ".gitignore", rooted=False,
                                    base=rel_dir)
            if sub and rel_dir:
                matchers.append(sub)
        keep = []
        for d in dirnames:
            rel = f"{rel_dir}/{d}" if rel_dir else d
            if d in ALWAYS_EXCLUDE or ignored(rel, True):
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in filenames:
            rel = f"{rel_dir}/{fn}" if rel_dir else fn
            if not ignored(rel, False):
                results.append(rel)
    return results


def enumerate_files(project: ProjectConfig) -> List[FileEntry]:
    root = project.root
    paths = _git_enumerate(root) if project.sync.respect_gitignore else None
    if paths is None:
        paths = _walk_enumerate(root, project.sync.respect_gitignore)

    extra_matchers: List[IgnoreMatcher] = []
    if project.sync.respect_dockerignore:
        m = _load_ignore_file(root / ".dockerignore", rooted=True, base="")
        if m:
            extra_matchers.append(m)
    if project.sync.exclude:
        extra_matchers.append(IgnoreMatcher(project.sync.exclude, rooted=False))

    entries: List[FileEntry] = []
    for rel in paths:
        top = rel.split("/", 1)[0]
        if top in ALWAYS_EXCLUDE:
            continue
        excluded = False
        for m in extra_matchers:
            # check the file and every parent dir against the matcher
            v = m.match(rel, False)
            parts = rel.split("/")
            for i in range(1, len(parts)):
                pv = m.match("/".join(parts[:i]), True)
                if pv is not None:
                    v = pv if v is None or pv else v
            if v:
                excluded = True
                break
        if excluded:
            continue
        full = root / rel
        try:
            st = full.lstat()
        except OSError:
            continue  # racy delete / dangling entry
        import stat as stat_mod
        if stat_mod.S_ISDIR(st.st_mode):
            continue
        entries.append(FileEntry(rel, st.st_size, stat_mod.S_ISLNK(st.st_mode)))
    return entries


# ---------------------------------------------------------------------------
# Big-item gate
# ---------------------------------------------------------------------------
def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}TB"


def apply_big_gate(project: ProjectConfig, entries: List[FileEntry], *,
                   allow_all: bool = False, reask: bool = False) -> List[FileEntry]:
    """Depth-1 gate: aggregate per top-level dir (and per root-level file).
    Items over the size/count threshold need a recorded yes/no. A 'yes' covers
    the whole directory — subdirectories are never asked about individually."""
    size_thresh = int(project.sync.big_dir_gb * (1 << 30))
    file_thresh = project.sync.big_dir_files

    groups: Dict[str, Tuple[int, int, bool]] = {}  # top -> (bytes, files, is_dir)
    for e in entries:
        parts = e.relpath.split("/", 1)
        top, is_dir = (parts[0], len(parts) > 1)
        b, c, d = groups.get(top, (0, 0, is_dir))
        groups[top] = (b + e.size, c + 1, d or is_dir)

    state = load_state(project, SYNC_STATE_FILE)
    decisions: Dict[str, Dict] = state.get("decisions", {})
    changed = False
    denied: List[str] = []
    pending: List[Tuple[str, int, int, bool]] = []

    for top, (nbytes, nfiles, is_dir) in sorted(groups.items()):
        big = nbytes >= size_thresh or nfiles >= file_thresh
        prev = decisions.get(top)
        if prev is not None and not reask:
            grown = nbytes > prev.get("size_bytes", 0) * REASK_GROWTH_FACTOR
            if not big or (not grown):
                if not prev.get("allow", False):
                    denied.append(top)
                continue
            # fell through: approved before but grew a lot -> re-ask
        if not big:
            continue
        if allow_all:
            decisions[top] = {"allow": True, "size_bytes": nbytes, "files": nfiles,
                              "decided_at": time.strftime("%Y-%m-%d %H:%M:%S"), "via": "--allow-large"}
            changed = True
            continue
        pending.append((top, nbytes, nfiles, is_dir))

    if pending:
        if not is_interactive():
            listing = "\n".join(
                f"  {t}{'/' if d else ''}  ({_human(b)}, {c} files)" for t, b, c, d in pending)
            raise SyncGateError(
                "Large items need a one-time decision before syncing:\n" + listing +
                "\nResolve by either:\n"
                "  - running `sw sync --review` interactively, or\n"
                "  - adding the item to sync.exclude in sw.yaml, or\n"
                "  - shipping it once as a dataset: `sw data push <dir>`, or\n"
                "  - passing --allow-large to approve everything this run."
            )
        for top, nbytes, nfiles, is_dir in pending:
            kind = "directory" if is_dir else "file"
            print(f"\n[sw] Large {kind}: {top}{'/' if is_dir else ''}  ({_human(nbytes)}, {nfiles} files)")
            print("     [y] sync it (remembered; whole dir, no per-subdir questions)")
            print("     [n] exclude it (remembered)")
            print("     [d] exclude it — and consider `sw data push` to mount it read-only instead")
            while True:
                ans = input("     sync this? [y/n/d]: ").strip().lower()
                if ans in ("y", "n", "d"):
                    break
            allow = ans == "y"
            decisions[top] = {"allow": allow, "size_bytes": nbytes, "files": nfiles,
                              "decided_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            changed = True
            if ans == "d":
                print(f"     -> excluded. Ship it once with: sw data push {top}")

    if changed:
        state["decisions"] = decisions
        save_state(project, SYNC_STATE_FILE, state)

    denied = [t for t, d in decisions.items() if not d.get("allow", False)]
    if denied:
        denied_set = set(denied)
        entries = [e for e in entries if e.relpath.split("/", 1)[0] not in denied_set]
    return entries


# ---------------------------------------------------------------------------
# Content hashing (with mtime-keyed cache so repeat runs are instant)
# ---------------------------------------------------------------------------
def _hash_file(path: Path, is_symlink: bool) -> str:
    h = hashlib.sha256()
    if is_symlink:
        h.update(b"symlink:" + os.readlink(path).encode())
        return h.hexdigest()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_code_hash(project: ProjectConfig, entries: List[FileEntry]) -> str:
    cache_path = project.state_dir / HASH_CACHE_FILE
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    new_cache = {}
    combined = hashlib.sha256()
    for e in sorted(entries, key=lambda x: x.relpath):
        full = project.root / e.relpath
        st = full.lstat()
        key = e.relpath
        cached = cache.get(key)
        if cached and cached[0] == st.st_size and cached[1] == st.st_mtime_ns:
            sha = cached[2]
        else:
            sha = _hash_file(full, e.is_symlink)
        new_cache[key] = [st.st_size, st.st_mtime_ns, sha]
        combined.update(f"{key}\0{sha}\n".encode())
    try:
        cache_path.write_text(json.dumps(new_cache))
    except OSError:
        pass
    return combined.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Snapshot push
# ---------------------------------------------------------------------------
def push_snapshot(project: ProjectConfig, cluster: ClusterConfig,
                  entries: List[FileEntry], code_hash: str, *, dry_run: bool = False) -> str:
    remote_dir = f"{cluster.project_dir(project.name)}/code/{code_hash}"
    marker = f"{remote_dir}/.sw_complete"
    total = sum(e.size for e in entries)
    if dry_run:
        info(f"[dry-run] would sync {len(entries)} files ({_human(total)}) -> {cluster.login}:{remote_dir}")
        return remote_dir
    if ssh_test(cluster.login, f"-f {marker}"):
        info(f"code snapshot {code_hash} already on {cluster.name} ({len(entries)} files) — skipping sync")
        return remote_dir
    ssh_mkdirs(cluster.login, [remote_dir])
    info(f"syncing {len(entries)} files ({_human(total)}) -> {cluster.login}:{remote_dir}")
    with tempfile.NamedTemporaryFile("w", suffix=".list", delete=False) as f:
        for e in entries:
            f.write(e.relpath + "\0")
        list_file = f.name
    try:
        args = ["-rlpt", "--from0", f"--files-from={list_file}"]
        if project.sync.follow_symlinks:
            args[0] = "-rLpt"
        rsync(str(project.root) + "/", f"{cluster.login}:{remote_dir}/", args, quiet=True)
        ssh_run(cluster.login, f"touch {marker}")
    finally:
        os.unlink(list_file)
    return remote_dir


def prune_snapshots(project: ProjectConfig, cluster: ClusterConfig, keep: int = 5) -> None:
    code_root = f"{cluster.project_dir(project.name)}/code"
    script = (
        f"cd {code_root} 2>/dev/null || exit 0; "
        f"ls -1t | tail -n +{keep + 1} | while read -r d; do "
        f"echo \"removing $d\"; rm -rf -- \"$d\"; done"
    )
    out = ssh_run(cluster.login, script).stdout.strip()
    if out:
        for line in out.splitlines():
            info(f"prune: {line}")
    else:
        info(f"prune: nothing to remove (<= {keep} snapshots)")
