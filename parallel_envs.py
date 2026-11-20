"""Step a batch of environments in worker processes.

ParallelEnvs holds len(seeds) environments split across `workers` processes.
rollouts() in reinforce.py accepts either a plain list of environments
(stepped in this process) or a ParallelEnvs; both produce identical episodes
for the same seeds. Use it when simulation dominates (MuJoCo, many CPU cores).
"""
import multiprocessing as mp

import numpy as np

from envs import make_env

KEEP_INFO = ("true_reward",)


def _worker(remote, env_id, seeds, env_kwargs):
    envs = [make_env(env_id, s, env_kwargs) for s in seeds]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send([env.reset()[0] for env in envs])
            elif cmd == "step":
                out = []
                for i, action in data:
                    obs, r, term, trunc, info = envs[i].step(action)
                    out.append((obs, r, term, trunc, {k: info[k] for k in KEEP_INFO if k in info}))
                remote.send(out)
            elif cmd == "close":
                break
    finally:
        remote.close()


class ParallelEnvs:
    def __init__(self, env_id, seeds, env_kwargs=None, workers=4):
        ctx = mp.get_context("spawn")
        workers = max(1, min(workers, len(seeds)))
        self.chunks = np.array_split(np.arange(len(seeds)), workers)
        self.owner = {}  # global env index -> (worker, local index)
        self.remotes, self.procs = [], []
        for w, chunk in enumerate(self.chunks):
            for local, i in enumerate(chunk):
                self.owner[int(i)] = (w, local)
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, env_id, [int(seeds[i]) for i in chunk], env_kwargs),
                               daemon=True)
            proc.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(proc)
        self.n = len(seeds)

    def __len__(self):
        return self.n

    def reset_all(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        out = [None] * self.n
        for w, remote in enumerate(self.remotes):
            for local, obs in enumerate(remote.recv()):
                out[int(self.chunks[w][local])] = obs
        return out

    def step(self, pairs):
        """pairs: [(global env index, action)] -> results in the same order."""
        per_worker = [[] for _ in self.remotes]
        order = [[] for _ in self.remotes]
        for k, (i, action) in enumerate(pairs):
            w, local = self.owner[i]
            per_worker[w].append((local, action))
            order[w].append(k)
        for w, remote in enumerate(self.remotes):
            if per_worker[w]:
                remote.send(("step", per_worker[w]))
        out = [None] * len(pairs)
        for w, remote in enumerate(self.remotes):
            if per_worker[w]:
                for k, result in zip(order[w], remote.recv()):
                    out[k] = result
        return out

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
