"""Provenance: git pinning of the synced code + the W&B paste-in snippet."""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slurm_wrangler.config import ConfigError, ProjectConfig
from slurm_wrangler.provenance import browse_url, git_provenance, redact_env, repro_command
from slurm_wrangler.sync import enumerate_files

SNIPPET = Path(__file__).parent.parent / "src/slurm_wrangler/templates/sw_wandb.py"
ID = ["-c", "user.name=t", "-c", "user.email=t@t"]


def _g(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _write(root, files):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


@pytest.fixture
def repo(tmp_path):
    """A project repo with a bare 'origin'; HEAD pushed to origin/main."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    root = tmp_path / "proj"
    root.mkdir()
    _write(root, {"sw.yaml": yaml.safe_dump({"name": "proj"}), "train.py": "print(1)\n",
                  "lib/a.py": "A\n", "gone.py": "x\n", ".gitignore": "/.sw\n"})
    _g(root, "init", "-q", "-b", "main")
    _g(root, *ID, "add", "-A")
    _g(root, *ID, "commit", "-qm", "init")
    _g(root, "remote", "add", "origin", str(remote))
    _g(root, "push", "-q", "-u", "origin", "main")
    return ProjectConfig.load(root), remote


def _prov(proj, **kw):
    entries = enumerate_files(proj)
    return git_provenance(proj.root, proj.state_dir, [e.relpath for e in entries], **kw)


def _autopush_refs(remote):
    return _g(remote, "for-each-ref", "refs/heads/autopush").splitlines()


class TestGitProvenance:
    def test_clean_pushed_head_no_push(self, repo):
        proj, remote = repo
        rec = _prov(proj)
        assert rec["commit"] == _g(proj.root, "rev-parse", "HEAD")
        assert not rec["dirty"] and rec["pushed"] and rec["autopush_branch"] is None
        assert _autopush_refs(remote) == []

    def test_dirty_tree_autopushes_exact_synced_state(self, repo):
        proj, remote = repo
        root = proj.root
        (root / "train.py").write_text("print(2)\n")           # modified
        (root / "new.py").write_text("new\n")                   # untracked -> synced
        (root / "gone.py").unlink()                             # deleted
        _write(root, {".venv/junk": "j"})                       # untracked, excluded below
        (root / "sw.yaml").write_text(yaml.safe_dump({"name": "proj", "sync": {"exclude": ["/.venv"]}}))
        proj = ProjectConfig.load(root)
        head, status_before = _g(root, "rev-parse", "HEAD"), _g(root, "status", "--porcelain")
        rec = _prov(proj, job_name="j")
        assert rec["dirty"] and rec["pushed"] and rec["autopush_branch"].startswith("autopush/")
        c = rec["commit"]
        assert _g(remote, "rev-parse", f"refs/heads/{rec['autopush_branch']}") == c
        files = set(_g(remote, "ls-tree", "-r", "--name-only", c).splitlines())
        assert {"train.py", "new.py", "lib/a.py", "sw.yaml"} <= files
        assert "gone.py" not in files and not any(f.startswith(".venv") for f in files)
        assert _g(remote, "show", f"{c}:train.py") == "print(2)"
        assert _g(remote, "rev-parse", f"{c}^") == head
        # the user's checkout is untouched
        assert _g(root, "rev-parse", "HEAD") == head
        assert _g(root, "status", "--porcelain") == status_before

    def test_same_state_pushed_once(self, repo):
        proj, remote = repo
        (proj.root / "train.py").write_text("print(3)\n")
        a, b = _prov(proj), _prov(proj)
        assert a["commit"] == b["commit"] and a["autopush_branch"] == b["autopush_branch"]
        assert len(_autopush_refs(remote)) == 1

    def test_unpushed_clean_head_is_pushed(self, repo):
        proj, remote = repo
        (proj.root / "train.py").write_text("print(4)\n")
        _g(proj.root, *ID, "commit", "-qam", "local")
        rec = _prov(proj)
        assert not rec["dirty"] and rec["commit"] == _g(proj.root, "rev-parse", "HEAD")
        assert _g(remote, "rev-parse", f"refs/heads/{rec['autopush_branch']}") == rec["commit"]

    def test_dry_run_and_no_push(self, repo):
        proj, remote = repo
        (proj.root / "train.py").write_text("print(5)\n")
        assert not _prov(proj, dry_run=True)["pushed"]
        assert not _prov(proj, push=False)["pushed"]
        assert _autopush_refs(remote) == []

    def test_project_in_subdir(self, repo):
        proj, remote = repo
        sub = proj.root / "lib"
        (sub / "sw.yaml").write_text("name: sub\n")
        _g(proj.root, *ID, "add", "-A")
        _g(proj.root, *ID, "commit", "-qm", "sub")
        _g(proj.root, "push", "-q")
        sproj = ProjectConfig.load(sub)
        (sub / "a.py").write_text("B\n")
        rec = _prov(sproj)
        assert _g(remote, "show", f"{rec['commit']}:lib/a.py") == "B"
        assert _g(remote, "show", f"{rec['commit']}:train.py") == "print(1)"

    def test_push_failure_does_not_raise(self, repo):
        proj, remote = repo
        _g(proj.root, "remote", "set-url", "origin", str(remote) + "-missing")
        (proj.root / "train.py").write_text("print(6)\n")
        rec = _prov(proj)
        assert not rec["pushed"] and rec["note"].startswith("push failed")

    def test_non_git(self, tmp_path):
        _write(tmp_path, {"sw.yaml": "name: p\n", "a.py": "x"})
        proj = ProjectConfig.load(tmp_path)
        assert git_provenance(proj.root, proj.state_dir, ["a.py"]) == {}


class TestHelpers:
    def test_redact(self):
        r = redact_env({"WANDB_API_KEY": "s", "HF_TOKEN": "t", "WANDB_PROJECT": "p"})
        assert r == {"WANDB_API_KEY": "<redacted>", "HF_TOKEN": "<redacted>", "WANDB_PROJECT": "p"}

    def test_urls_and_repro(self):
        assert browse_url("git@github.com:o/r.git", "abc") == "https://github.com/o/r/tree/abc"
        assert browse_url("https://github.com/o/r.git", "abc") == "https://github.com/o/r/tree/abc"
        cmd = repro_command({"commit": "abc", "remote": "origin", "autopush_branch": "autopush/x"},
                            ["run", "--gpus", "1", "--", "python", "t.py", "--x", "a b"])
        assert cmd == ("git fetch origin autopush/x && git checkout abc && "
                       "sw run --gpus 1 -- python t.py --x 'a b'")

    def test_provenance_config(self, tmp_path):
        _write(tmp_path, {"sw.yaml": yaml.safe_dump({"name": "p", "provenance": {"git": "off", "remote": "up"}})})
        p = ProjectConfig.load(tmp_path)
        assert p.provenance.git == "off" and p.provenance.remote == "up"
        (tmp_path / "sw.yaml").write_text(yaml.safe_dump({"name": "p", "provenance": {"git": "maybe"}}))
        with pytest.raises(ConfigError):
            ProjectConfig.load(tmp_path)

    def test_strip_dry_run(self):
        from slurm_wrangler.cli import _strip_dry_run
        assert _strip_dry_run(["run", "--dry-run", "--", "x", "--dry-run"]) == ["run", "--", "x", "--dry-run"]


class TestWandbSnippet:
    def test_noop_without_meta_dir(self, monkeypatch):
        monkeypatch.delenv("SW_META_DIR", raising=False)
        exec(compile(SNIPPET.read_text(), "sw_wandb", "exec"), {})

    def test_wraps_init_and_uploads(self, tmp_path, monkeypatch):
        (tmp_path / "sw_meta.json").write_text(json.dumps(
            {"git": {"commit": "abc", "autopush_branch": "autopush/x"}, "repro": "r"}))
        (tmp_path / "sbatch.sh").write_text("#!/bin/bash\n")
        saved, cfg = [], {}

        class Run:
            def __init__(self):
                self.id = "r1"
                self.config = types.SimpleNamespace(update=lambda d, **k: cfg.update(d))
                self.summary = {}

            def save(self, path, base_path=None, policy=None):
                saved.append(os.path.relpath(path, base_path))

        fake = types.ModuleType("wandb")
        fake.run = None
        fake.init = lambda *a, **k: Run()
        monkeypatch.setitem(sys.modules, "wandb", fake)
        monkeypatch.setenv("SW_META_DIR", str(tmp_path))
        monkeypatch.setenv("SLURM_JOB_ID", "42")
        exec(compile(SNIPPET.read_text(), "sw_wandb", "exec"), {})
        exec(compile(SNIPPET.read_text(), "sw_wandb", "exec"), {})  # pasted twice: wraps once
        run = fake.init(project="p")
        assert sorted(saved) == ["sbatch.sh", "sw_meta.json"]
        assert cfg["sw"]["git"]["commit"] == "abc" and cfg["sw"]["slurm_job_id"] == "42"
        assert run.summary["sw/commit"] == "abc" and run.summary["sw/branch"] == "autopush/x"


class TestSubmitMetadata:
    """Real submit() + real runtime/submit_slurm.sh, run locally (ssh/image/rsync faked)."""

    RUNTIME = Path(__file__).parent.parent / "src/slurm_wrangler/runtime"

    def _setup(self, repo, tmp_path, monkeypatch, sbatch_body):
        from slurm_wrangler import submit as sub
        from slurm_wrangler.config import ClusterConfig, GlobalConfig
        proj, _ = repo
        (proj.root / "sw.yaml").write_text(yaml.safe_dump({"name": "proj", "defaults": {"targets": [
            {"partition": "gpu-a40", "account": "lab"}, {"partition": "ckpt", "account": "ckpt-lab"}]}}))
        proj = ProjectConfig.load(proj.root)
        (proj.root / "train.py").write_text("print('edited')\n")   # dirty -> autopush
        store = tmp_path / "store"
        cluster = ClusterConfig(name="hyak", login="fake", storage_root=str(store), mount_dir="/mmfs1")
        gcfg = GlobalConfig(user="u", clusters={"hyak": cluster}, default_cluster="hyak", forward_env=[])
        bindir = tmp_path / "bin"
        bindir.mkdir()
        (bindir / "sbatch").write_text("#!/bin/bash\n" + sbatch_body)
        (bindir / "sbatch").chmod(0o755)
        monkeypatch.setattr(sub, "ensure_image", lambda *a, **k: "/x.sif")
        monkeypatch.setattr(sub, "push_snapshot", lambda *a, **k: "/code")
        monkeypatch.setattr(sub, "ship_runtime", lambda *a, **k: str(self.RUNTIME))
        monkeypatch.setattr(sub, "ssh_mkdirs",
                            lambda login, ds: [Path(d).mkdir(parents=True, exist_ok=True) for d in ds])
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", TMPDIR=str(tmp_path))
        monkeypatch.setattr(sub, "ssh_run", lambda login, cmd, **k: subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True, env=env, check=True))
        return sub, proj, gcfg, store

    @staticmethod
    def _runs(store):
        return sorted((store / "projects/proj/runs").iterdir())

    def test_race_writes_one_meta_dir_per_target(self, repo, tmp_path, monkeypatch):
        n = tmp_path / "n"
        sub, proj, gcfg, store = self._setup(
            repo, tmp_path, monkeypatch,
            f'echo $(( $(cat {n} 2>/dev/null || echo 100) + 1 )) > {n}; echo "Submitted batch job $(cat {n})"\n')
        argv = ["run", "--race", "--", "python", "train.py"]
        sub.submit(proj, gcfg, sub.SubmitParams(race=True), ["python", "train.py"], argv=argv)
        runs = self._runs(store)
        assert [r.name[-3:] for r in runs] == ["-t0", "-t1"]
        metas = [json.loads((r / "sw_meta.json").read_text()) for r in runs]
        assert [m["slurm"]["partition"] for m in metas] == ["gpu-a40", "ckpt"]
        assert [m["slurm"]["account"] for m in metas] == ["lab", "ckpt-lab"]
        assert metas[0]["git"]["commit"] == metas[1]["git"]["commit"]       # one pinned commit
        assert metas[0]["git"]["autopush_branch"]
        assert all(m["sw_env"]["SW_RACE_GROUP"] == m["job_name"] for m in metas)
        assert all(m["meta_dir"] == str(r) for m, r in zip(metas, runs))
        assert [(r / "slurm_job_id").read_text().strip() for r in runs] == ["101", "102"]
        for r, part in zip(runs, ["gpu-a40", "ckpt"]):
            assert f"#SBATCH --partition={part}" in (r / "sbatch.sh").read_text()
        recs = [json.loads(line) for line in (proj.state_dir / "jobs.jsonl").read_text().splitlines()]
        assert [rec["meta_dir"] for rec in recs] == [str(r) for r in runs]

    def test_race_target_skipped_after_claim_writes_nothing(self, repo, tmp_path, monkeypatch):
        # target 0 "starts" instantly and claims the group -> target 1 must not submit or write meta
        sub, proj, gcfg, store = self._setup(
            repo, tmp_path, monkeypatch,
            'd="$SW_SLURM_LOGS_DIR/.race_${SW_RACE_GROUP}"; mkdir -p "$d"; echo 555 > "$d/owner"\n'
            'echo "Submitted batch job 555"\n')
        sub.submit(proj, gcfg, sub.SubmitParams(race=True), ["python", "train.py"])
        runs = self._runs(store)
        assert [r.name[-3:] for r in runs] == ["-t0"]
        assert (runs[0] / "slurm_job_id").read_text().strip() == "555"
        assert len((proj.state_dir / "jobs.jsonl").read_text().splitlines()) == 1

    def test_fixed_name_gets_distinct_dirs(self, repo, tmp_path, monkeypatch):
        sub, proj, gcfg, store = self._setup(repo, tmp_path, monkeypatch, 'echo "Submitted batch job 7"\n')
        sub.submit(proj, gcfg, sub.SubmitParams(name="fixed"), ["python", "train.py"])
        sub.submit(proj, gcfg, sub.SubmitParams(name="fixed"), ["python", "train.py"])
        runs = self._runs(store)
        assert len(runs) == 2 and all(r.name.startswith("fixed-") for r in runs)
