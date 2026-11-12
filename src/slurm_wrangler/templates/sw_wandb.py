# --- slurm_wrangler -> W&B run provenance (from `sw snippet wandb`) ----------------------
# Paste once near the top of your training entrypoint, after the imports. In an `sw run`
# job, every wandb run this process creates gets the pinned git commit, the generated
# sbatch script and the full sw invocation (sw_meta.json), no matter where wandb.init is
# called (your code or a library). Outside sw jobs, or without wandb, it does nothing.
def _sw_wandb_provenance():
    import functools
    import json
    import os

    meta_dir = os.environ.get("SW_META_DIR")
    if not meta_dir or not os.path.isfile(os.path.join(meta_dir, "sw_meta.json")):
        return
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb.init, "_sw_wrapped", False):
        return
    attached = set()

    def attach(run):
        if run is None or run.id in attached:
            return
        try:
            with open(os.path.join(meta_dir, "sw_meta.json")) as f:
                meta = json.load(f)
            meta["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
            run.config.update({"sw": meta}, allow_val_change=True)
            git = meta.get("git") or {}
            run.summary["sw/commit"] = git.get("commit")
            run.summary["sw/branch"] = git.get("autopush_branch") or git.get("branch")
            run.summary["sw/commit_url"] = git.get("commit_url")
            run.summary["sw/repro"] = meta.get("repro")
            for name in sorted(os.listdir(meta_dir)):
                if os.path.isfile(os.path.join(meta_dir, name)):
                    run.save(os.path.join(meta_dir, name), base_path=meta_dir, policy="now")
            attached.add(run.id)
        except Exception as e:  # noqa: BLE001  (provenance must never break training)
            print(f"[sw] W&B provenance upload failed: {e}")

    orig_init = wandb.init

    @functools.wraps(orig_init)
    def init(*args, **kwargs):
        run = orig_init(*args, **kwargs)
        attach(run)
        return run

    init._sw_wrapped = True
    wandb.init = init
    attach(wandb.run)


_sw_wandb_provenance()
# --- end slurm_wrangler provenance ---------------------------------------------------------
