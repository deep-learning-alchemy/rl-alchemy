"""Run configs on Modal. Used through launch(configs, modal=True).

Each run gets its own container. Environments simulate on the CPU, so most
runs use a CPU-only container; runs with device="cuda" (large networks) get an
H100 container instead. Runs write to the persistent Volume mounted at
/root/runs; metrics and configs are copied back to your local runs/.
"""
from pathlib import Path, PurePosixPath

import modal

from utils import REPO_ROOT, config_str, runs_dir

MODAL_ENVIRONMENT = config_str("CONFIG_MODAL_ENVIRONMENT", "MODAL_ENVIRONMENT")
if MODAL_ENVIRONMENT is None:  # never fall back to a default environment others may share
    raise RuntimeError("Set CONFIG_MODAL_ENVIRONMENT in utils.py to your own Modal environment "
                       "(or export MODAL_ENVIRONMENT) before using Modal.")
APP_NAME = config_str("CONFIG_MODAL_APP_NAME", default="rl_alchemy")
VOLUME_NAME = config_str("CONFIG_MODAL_VOLUME_NAME", "MODAL_VOLUME_NAME", default="volume-rl-alchemy")
WANDB_SECRET = config_str("CONFIG_MODAL_WANDB_SECRET", default="dl-alchemy-wandb")
REMOTE_REPO = PurePosixPath("/root/rl-alchemy")
REMOTE_RUNS = PurePosixPath("/root/runs")
IGNORED = {".git", ".venv", "__pycache__", "runs", "plots", "results", "wandb", "handout", "staff", "renders"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(uv_project_dir=str(REPO_ROOT))
    .env({"RL_ALCHEMY_RUNS_DIR": str(REMOTE_RUNS), "WANDB_DIR": "/tmp/wandb",
          "WANDB_ENTITY": config_str("CONFIG_WANDB_ENTITY", default=""),
          "WANDB_PROJECT": config_str("CONFIG_WANDB_PROJECT", default="rl-alchemy")})
    .workdir(str(REMOTE_REPO))
    .add_local_dir(REPO_ROOT, remote_path=str(REMOTE_REPO),
                   ignore=lambda p: any(part in IGNORED for part in Path(p).parts))
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True,
                                environment_name=MODAL_ENVIRONMENT)
app = modal.App(APP_NAME)


FUNCTION_ARGS = dict(image=image, timeout=6 * 3600, volumes={str(REMOTE_RUNS): volume},
                     secrets=[modal.Secret.from_name(WANDB_SECRET, environment_name=MODAL_ENVIRONMENT)]
                     if config_str("CONFIG_WANDB_ENTITY", "WANDB_ENTITY") else [])


def _run(cfg):
    import reinforce
    volume.reload()  # see checkpoints and datasets written by earlier stages
    run_dir = reinforce.run(cfg, verbose=True)
    volume.commit()
    return cfg.name(), {name: (run_dir / name).read_text()
                        for name in ("config.json", "metrics.jsonl", "summary.json", "DONE")
                        if (run_dir / name).exists()}


@app.function(cpu=2.0, memory=4096, **FUNCTION_ARGS)
def run_cpu(cfg):
    return _run(cfg)


@app.function(cpu=8.0, memory=8192, **FUNCTION_ARGS)
def run_cpu8(cfg):
    """CPU runs that step environments in parallel (env_workers > 1)."""
    return _run(cfg)


@app.function(gpu="H100", cpu=8.0, memory=32768, **FUNCTION_ARGS)
def run_gpu(cfg):
    return _run(cfg)


def _function_for(cfg):
    if getattr(cfg, "device", "cpu") != "cpu":
        return run_gpu
    return run_cpu8 if getattr(cfg, "env_workers", 1) > 1 else run_cpu


def launch_modal(configs, detach=False):
    """Run configs on Modal. detach=True spawns every run and returns at once;
    the runs keep going if this machine disconnects, and their results stay on
    the Volume (fetch them with fetch_modal or `modal volume get`)."""
    if not MODAL_ENVIRONMENT:
        raise RuntimeError("Set CONFIG_MODAL_ENVIRONMENT in utils.py first.")
    from launch import stages
    if detach:
        groups = stages(configs)
        if len(groups) > 1:
            raise ValueError("detach=True needs runs without dependencies; launch each stage separately")
        with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT, detach=True):
            calls = [(cfg.name(), _function_for(cfg).spawn(cfg)) for cfg in configs]
        for name, call in calls:
            print("spawned", name, call.object_id, flush=True)
        return calls
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        for stage in stages(configs):
            if not stage:
                continue
            groups = {}
            for cfg in stage:
                groups.setdefault(_function_for(cfg), []).append(cfg)
            calls = [fn.map(group, return_exceptions=True, order_outputs=False) for fn, group in groups.items()]
            for result in (r for call in calls for r in call):
                if isinstance(result, Exception):
                    print("FAILED:", repr(result), flush=True)
                    continue
                name, files = result
                local = runs_dir() / name
                local.mkdir(parents=True, exist_ok=True)
                for filename, text in files.items():
                    (local / filename).write_text(text)
                print("finished", name, flush=True)


def fetch_modal(names):
    """Copy finished runs' metrics and configs from the Volume into local runs/."""
    import subprocess
    done = []
    for name in names:
        local = runs_dir() / name
        local.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "metrics.jsonl", "DONE"):
            proc = subprocess.run(["modal", "volume", "get", "--force", "--env", MODAL_ENVIRONMENT, VOLUME_NAME,
                                   f"{name}/{filename}", str(local / filename)], capture_output=True, text=True)
            if filename == "DONE" and proc.returncode == 0:
                done.append(name)
    return done


def fetch_files(run, files=("config.json", "pi_final.pt")):
    """Copy specific files of a run (e.g. checkpoints, datasets) from the Volume into runs/<run>/."""
    vol = modal.Volume.from_name(VOLUME_NAME, environment_name=MODAL_ENVIRONMENT)
    local = runs_dir() / run
    local.mkdir(parents=True, exist_ok=True)
    for name in files:
        (local / name).write_bytes(b"".join(vol.read_file(f"{run}/{name}")))
