"""Watch a saved policy: render an episode to a GIF with a camera that follows the robot.

    uv run python -m render RUN [RUN ...]                     # runs/<RUN>/pi_final.pt
    uv run python -m render RUN --file pi_1000000.pt          # a checkpoint (save_checkpoints)
    uv run python -m render RUN --fetch                       # copy it from the Modal Volume first
    uv run python -m render RUN --steps 1000 --greedy         # longer; mean action, no noise

GIFs are written to renders/<RUN>[-<checkpoint>].gif. The episode uses the
run's own env_kwargs (for example, no fall rule), unless --env overrides them
with JSON, e.g. --env '{"terminate_when_unhealthy": true}'. Rendering runs on
your machine; on Linux without a display set MUJOCO_GL=egl (or osmesa).
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from reinforce import load_policy

OUT = Path(__file__).parent / "renders"


def render(run, file="pi_final.pt", steps=600, every=2, greedy=False, seed=0, env_kwargs=None, out=OUT):
    """Roll out one episode and save it as a GIF. Returns a summary dict."""
    policy, cfg = load_policy(run if file == "pi_final.pt" else f"{run}/{file}")
    kwargs = {**cfg.env_kwargs, **dict(env_kwargs or {})}
    env = gym.make(cfg.env_id, render_mode="rgb_array", width=480, height=360, camera_name="track", **kwargs)
    obs, _ = env.reset(seed=seed)
    gen = torch.Generator().manual_seed(seed)
    x0 = float(env.unwrapped.data.qpos[0])
    frames, proxy, forward, fell, t = [], 0.0, 0.0, False, 0
    for t in range(steps):
        with torch.no_grad():
            a, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32)[None], gen, greedy=greedy)
        obs, r, term, trunc, info = env.step(np.clip(a[0].numpy(), env.action_space.low, env.action_space.high))
        proxy += r
        forward += info.get("x_velocity", 0.0)
        if t % every == 0:
            frames.append(Image.fromarray(env.render()))
        if term or trunc:
            fell = bool(term)
            break
    frames += [frames[-1]] * 10  # hold the last frame so a fall is visible
    out.mkdir(exist_ok=True)
    name = f"{run}{'' if file == 'pi_final.pt' else '-' + file[:-3]}{'-greedy' if greedy else ''}"
    path = out / f"{name}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * env.unwrapped.dt * every), loop=0)
    summary = dict(gif=str(path), steps=t + 1, fell=fell, proxy_return=proxy, forward_progress=forward,
                   distance_m=float(env.unwrapped.data.qpos[0]) - x0)
    env.close()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--file", default="pi_final.pt", help="pi_final.pt or a checkpoint such as pi_1000000.pt")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--greedy", action="store_true", help="act with the mean action")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", default=None, help="JSON env_kwargs overriding the run's")
    p.add_argument("--fetch", action="store_true", help="copy the run's config and policy from the Modal Volume")
    args = p.parse_args()
    for run in args.runs:
        if args.fetch:
            from modal_launch import fetch_files
            fetch_files(run, ["config.json", args.file])
        s = render(run, args.file, args.steps, greedy=args.greedy, seed=args.seed,
                   env_kwargs=json.loads(args.env) if args.env else None)
        print(f"{run} ({args.file}): {s['steps']} steps, {'fell' if s['fell'] else 'did not fall'}, "
              f"{s['distance_m']:.1f} m forward -> {s['gif']}", flush=True)


if __name__ == "__main__":
    main()
