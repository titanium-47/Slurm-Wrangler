"""Unit tests for slurm_wrangler core logic (no cluster access needed)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slurm_wrangler.config import GlobalConfig, ProjectConfig, ConfigError
from slurm_wrangler.sync import (IgnoreMatcher, enumerate_files, apply_big_gate,
                                 compute_code_hash, SyncGateError)
from slurm_wrangler.image import env_hash, _generated_dockerfile
from slurm_wrangler.presets import expand_cache_entries
from slurm_wrangler.submit import (SubmitParams, build_static_binds, encode_caches,
                                   compute_mem_directive, collect_forward_env, runtime_hash)


# ---------------------------------------------------------------------------
def make_project(tmp_path, raw=None, files=None):
    raw = raw or {"name": "proj"}
    for rel, content in (files or {}).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, int):  # size in bytes
            with open(p, "wb") as f:
                f.seek(content - 1)
                f.write(b"\0")
        else:
            p.write_text(content)
    (tmp_path / "sw.yaml").write_text(yaml.safe_dump(raw))
    return ProjectConfig.load(tmp_path)


def git_init(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)


# ---------------------------------------------------------------------------
class TestIgnoreMatcher:
    def test_basic(self):
        m = IgnoreMatcher(["*.pyc", "logs/", "/rooted.txt"])
        assert m.match("a/b/x.pyc", False) is True
        assert m.match("logs", True) is True
        assert m.match("rooted.txt", False) is True
        assert m.match("sub/rooted.txt", False) is None
        assert m.match("keep.py", False) is None

    def test_negation(self):
        m = IgnoreMatcher(["*.log", "!important.log"])
        assert m.match("x.log", False) is True
        assert m.match("important.log", False) is False

    def test_doublestar(self):
        m = IgnoreMatcher(["docs/**/tmp"])
        assert m.match("docs/a/b/tmp", False) is True

    def test_rooted_dockerignore(self):
        m = IgnoreMatcher(["node_modules"], rooted=True)
        assert m.match("node_modules", True) is True
        assert m.match("sub/node_modules", True) is None


class TestEnumerate:
    def test_git_respects_gitignore(self, tmp_path):
        proj = make_project(tmp_path, files={
            "keep.py": "x", "skip.log": "y", ".gitignore": "*.log\n",
            "sub/inner.py": "z", "wandb/run.txt": "w",
        })
        (tmp_path / ".gitignore").write_text("*.log\nwandb/\n")
        git_init(tmp_path)
        rels = {e.relpath for e in enumerate_files(proj)}
        assert "keep.py" in rels and "sub/inner.py" in rels
        assert "skip.log" not in rels and "wandb/run.txt" not in rels
        assert not any(r.startswith(".git/") for r in rels)

    def test_nongit_walk(self, tmp_path):
        proj = make_project(tmp_path, files={
            "keep.py": "x", "skip.log": "y", ".gitignore": "*.log\n",
        })
        rels = {e.relpath for e in enumerate_files(proj)}
        assert "keep.py" in rels and "skip.log" not in rels

    def test_dockerignore_and_extra_excludes(self, tmp_path):
        proj = make_project(tmp_path,
            raw={"name": "proj", "sync": {"exclude": ["*.ckpt"]}},
            files={"a.py": "x", "big.ckpt": "y", ".dockerignore": "secret/\n",
                   "secret/k.txt": "s"})
        git_init(tmp_path)
        rels = {e.relpath for e in enumerate_files(proj)}
        assert "a.py" in rels
        assert "big.ckpt" not in rels
        assert "secret/k.txt" not in rels


class TestBigGate:
    def test_gate_blocks_noninteractive(self, tmp_path, monkeypatch):
        proj = make_project(tmp_path, raw={"name": "p", "sync": {"big_dir_gb": 0.000001}},
                            files={"bigdir/blob.bin": 2048, "small.py": "x"})
        monkeypatch.setattr("slurm_wrangler.sync.is_interactive", lambda: False)
        entries = enumerate_files(proj)
        with pytest.raises(SyncGateError):
            apply_big_gate(proj, entries)

    def test_allow_all_persists(self, tmp_path, monkeypatch):
        proj = make_project(tmp_path, raw={"name": "p", "sync": {"big_dir_gb": 0.000001}},
                            files={"bigdir/blob.bin": 2048, "small.py": "x"})
        monkeypatch.setattr("slurm_wrangler.sync.is_interactive", lambda: False)
        entries = enumerate_files(proj)
        kept = apply_big_gate(proj, entries, allow_all=True)
        assert any(e.relpath.startswith("bigdir/") for e in kept)
        # decision persisted -> second call needs no allow_all
        kept2 = apply_big_gate(proj, enumerate_files(proj))
        assert any(e.relpath.startswith("bigdir/") for e in kept2)

    def test_denied_dir_excluded(self, tmp_path, monkeypatch):
        proj = make_project(tmp_path, raw={"name": "p", "sync": {"big_dir_gb": 0.000001}},
                            files={"bigdir/blob.bin": 2048, "small.py": "x"})
        state = proj.state_dir / "sync.local.yaml"
        state.write_text(yaml.safe_dump({"decisions": {"bigdir": {
            "allow": False, "size_bytes": 2048, "files": 1}}}))
        monkeypatch.setattr("slurm_wrangler.sync.is_interactive", lambda: False)
        kept = apply_big_gate(proj, enumerate_files(proj))
        assert not any(e.relpath.startswith("bigdir/") for e in kept)
        assert any(e.relpath == "small.py" for e in kept)


class TestHash:
    def test_stable_and_sensitive(self, tmp_path):
        proj = make_project(tmp_path, files={"a.py": "aaa", "b/c.py": "ccc"})
        entries = enumerate_files(proj)
        h1 = compute_code_hash(proj, entries)
        h2 = compute_code_hash(proj, entries)  # cached path
        assert h1 == h2 and len(h1) == 12
        (tmp_path / "a.py").write_text("changed")
        h3 = compute_code_hash(proj, enumerate_files(proj))
        assert h3 != h1

    def test_env_hash_changes_with_deps(self, tmp_path):
        proj = make_project(tmp_path, raw={"name": "p"},
                            files={"environment.yml": "dependencies: [python=3.10]"})
        h1 = env_hash(proj)
        (tmp_path / "environment.yml").write_text("dependencies: [python=3.11]")
        h2 = env_hash(proj)
        assert h1 != h2 and len(h1) == 8

    def test_env_hash_missing_file(self, tmp_path):
        proj = make_project(tmp_path, raw={"name": "p"})
        with pytest.raises(ConfigError):
            env_hash(proj)


class TestPresets:
    def test_isaac_preset_expands(self):
        entries = expand_cache_entries([{"preset": "isaac-sim"}])
        assert len(entries) == 7
        assert any(e.container == "/isaac-sim/kit/cache" for e in entries)
        shared = [e for e in entries if e.host_slug == "isaac/data"]
        assert len(shared) == 2  # ov/data + kit/data share one host tree

    def test_custom_entry(self):
        (e,) = expand_cache_entries([{"container": "/root/.cache/foo", "writeback": True}])
        assert e.host_slug == "root-cache-foo"

    def test_unknown_preset(self):
        with pytest.raises(ConfigError):
            expand_cache_entries([{"preset": "nope"}])


class TestSubmitPieces:
    def _cluster(self):
        from slurm_wrangler.config import ClusterConfig
        return ClusterConfig(name="hyak", login="klone-login",
                             storage_root="/gscratch/weirdlab/u/sw", mount_dir="/mmfs1",
                             force_constraint="a40|a100",
                             slurm={"mem_mode": "per_node_from_gpu", "mem_per_gpu_gb": 60,
                                    "mem_cpu_job_gb": 32})

    def test_mem_directive(self):
        c = self._cluster()
        assert compute_mem_directive(c, SubmitParams(), 4, c.slurm) == "#SBATCH --mem=240G"
        assert compute_mem_directive(c, SubmitParams(mem="100G"), 4, c.slurm) == "#SBATCH --mem=100G"
        assert compute_mem_directive(c, SubmitParams(), 0, c.slurm) == "#SBATCH --mem=32G"
        c.slurm = {"mem_mode": "per_cpu", "mem_per_cpu_gb": 25}
        assert compute_mem_directive(c, SubmitParams(), 4, c.slurm) == "#SBATCH --mem-per-cpu=25G"

    def test_static_binds(self, tmp_path):
        proj = make_project(tmp_path, raw={
            "name": "p", "data": [{"name": "lt", "mount": "/data/lt"}],
            "persist": ["logs"]})
        binds = build_static_binds(proj, self._cluster())
        assert "/gscratch/weirdlab/u/sw/projects/p/persist/logs::/workspace/p/logs::rw" in binds
        assert "/gscratch/weirdlab/u/sw/projects/p/data/lt::/data/lt::ro" in binds
        assert "/mmfs1::/mmfs1::rw" in binds

    def test_forward_env(self, tmp_path, monkeypatch):
        proj = make_project(tmp_path, raw={"name": "p", "env": {"STATIC": "1"}})
        monkeypatch.setenv("WANDB_API_KEY", "secret")
        monkeypatch.setenv("UNRELATED", "no")
        g = GlobalConfig(user="u", forward_env=["WANDB_*"])
        fwd = collect_forward_env(g, proj, {"EXTRA": "2"})
        assert fwd["WANDB_API_KEY"] == "secret"
        assert fwd["STATIC"] == "1" and fwd["EXTRA"] == "2"
        assert "UNRELATED" not in fwd

    def test_runtime_hash_stable(self):
        assert runtime_hash() == runtime_hash()
        assert len(runtime_hash()) == 8

    def test_encode_caches_roundtrip(self, tmp_path):
        import base64
        proj = make_project(tmp_path, raw={"name": "p", "cache": [{"preset": "huggingface"}]})
        decoded = base64.b64decode(encode_caches(proj)).decode()
        assert decoded == "root-cache-huggingface|/root/.cache/huggingface|1|ignore-existing|"


class TestDockerfileGen:
    def test_generated(self, tmp_path):
        proj = make_project(tmp_path, raw={
            "name": "p", "image": {"kind": "conda", "environment": "environment.yml",
                                   "apt": ["ffmpeg"], "pip": ["wandb==0.17"]}},
            files={"environment.yml": "dependencies: [python=3.10]"})
        df = _generated_dockerfile(proj)
        assert "micromamba create -y -p /opt/env" in df
        assert "ffmpeg" in df and "wandb==0.17" in df
        assert "PATH=/opt/env/bin:$PATH" in df


class TestConfig:
    def test_project_defaults(self, tmp_path):
        proj = make_project(tmp_path, raw={"name": "my-proj"})
        assert proj.workdir == "/workspace/my-proj"
        assert proj.launcher == "plain"
        assert proj.persist == ["logs", "outputs"]

    def test_bad_launcher(self, tmp_path):
        with pytest.raises(ConfigError):
            make_project(tmp_path, raw={"name": "p", "launcher": "mpi"})

    def test_find_root_from_subdir(self, tmp_path):
        proj = make_project(tmp_path, files={"sub/deep/x.py": "1"})
        found = ProjectConfig.load(tmp_path / "sub" / "deep")
        assert found.root == tmp_path
