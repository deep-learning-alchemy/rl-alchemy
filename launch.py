"""Run lists of configs locally (CPU processes) or on Modal (CPU containers).

    from launch import launch
    launch(configs)                        # local, one process per CPU core
    launch(configs, workers=4)             # local, at most 4 at a time
    launch(configs, modal=True)            # Modal, one CPU container per run
    launch(configs, dry_run=True)          # print run names only

A default Humanoid run (5M steps, 8 environment workers) takes about an hour
on an 8-CPU Modal container; the widest policies need device="cuda". Runs whose metrics already
exist are skipped unless overwrite=True, so re-running a launcher after a crash
only fills in what is missing.

On Modal, runs write to the Volume named in utils.py and the metrics and
config of each finished run are copied back into your local runs/ directory.
Checkpoints stay on the Volume. Fetch them with
    uv run modal volume get volume-rl-alchemy runs/<run_name> runs/
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from reinforce import run
from utils import runs_dir


def _warn_same_config(configs):
    """Runs are deterministic given their seeds: configs that differ only in
    their names repeat the same computation."""
    import dataclasses
    ignored = {"run_name", "tags", "save_checkpoints"}
    seen = {}
    for cfg in configs:
        key = json.dumps({k: v for k, v in dataclasses.asdict(cfg).items() if k not in ignored},
                         sort_keys=True, default=str)
        if key in seen:
            print(f"note: {cfg.name()} repeats {seen[key]} (same config and seed)")
        else:
            seen[key] = cfg.name()


def _done(cfg):
    return (runs_dir() / cfg.name() / "DONE").exists()


def dependencies(cfg):
    """Run names this config reads from: initializations and references."""
    specs = [getattr(cfg, k, None) for k in ("init_from", "kl_ref")]
    return {spec.split("/")[0] for spec in specs if spec}


def stages(configs):
    """Group configs so that every run starts after the runs it reads from."""
    pending, done, out = list(configs), set(), []
    names = {c.name() for c in configs}
    while pending:
        ready = [c for c in pending if not (dependencies(c) & names) - done]
        if not ready:
            raise ValueError("circular run dependencies")
        out.append(ready)
        done |= {c.name() for c in ready}
        pending = [c for c in pending if c not in ready]
    return out


def _run_quiet(cfg):
    run(cfg, verbose=False)
    return cfg.name()


def launch(configs, *, modal=False, workers=None, dry_run=False, overwrite=False, detach=False):
    by_name = {}
    for cfg in configs:
        if by_name.setdefault(cfg.name(), cfg) != cfg:
            raise ValueError(f"two different configs share the run name {cfg.name()!r}")
    configs = list(by_name.values())  # identical duplicates run once
    _warn_same_config(configs)
    todo = [cfg for cfg in configs if overwrite or not _done(cfg)]
    print(f"{len(configs)} configs, {len(configs) - len(todo)} already finished, {len(todo)} to run")
    if dry_run:
        for cfg in todo:
            print("  ", cfg.name())
        return
    if not todo:
        return
    if modal:
        from modal_launch import launch_modal
        return launch_modal(todo, detach=detach)
    workers = workers or os.cpu_count()
    for stage in stages(todo):
        if not stage:
            continue
        with ProcessPoolExecutor(min(workers, len(stage))) as pool:
            futures = {pool.submit(_run_quiet, cfg): cfg.name() for cfg in stage}
            for i, future in enumerate(as_completed(futures), 1):
                name = futures[future]
                try:
                    future.result()
                    print(f"[{i}/{len(stage)}] finished {name}", flush=True)
                except Exception as error:  # keep the sweep going; report at the end
                    print(f"[{i}/{len(stage)}] FAILED {name}: {error!r}", flush=True)


def buffer_path(run_name):
    """A run's saved episodes, relative to runs/ so it resolves locally and on Modal."""
    return f"{run_name}/episodes.npz"


def summary(run_name):
    return json.loads((runs_dir() / run_name / "summary.json").read_text())
