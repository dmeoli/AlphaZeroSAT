"""Evaluation for AlphaZeroSAT (reproduces the paper's metric).

Measures the average number of branching decisions the trained policy needs to
solve a set of SAT problems --- the metric of Wang & Rompf (2018), Fig. 2. The
model is run greedily (MCTS with tau -> 0, each problem solved once); the env's
step counter gives the number of decisions.

Used both standalone and by train_torch.py to log decisions vs training cycle.
"""
import os
import time

import numpy as np

from mct import MCT


def evaluate_decisions(predict, path, n_files, max_clause=120, max_var=20,
                       n_batch=16, resign=400, limit=None, with_time=False):
    """Mean branching decisions to solve the problems in `path`.

    `predict(states) -> (policy_logits, value)` is the network forward (e.g.
    AZTrainer.predict). Returns the mean over the evaluated problems (lower is
    better; compare to MiniSat / across training checkpoints). With `with_time`,
    returns ``(mean_decisions, mean_seconds_per_problem)`` --- each decision runs an
    MCTS search, so the wall-clock cost is far higher than MiniSat's per instance."""
    t0 = time.perf_counter()
    n_total = len([f for f in os.listdir(path) if f.endswith(".cnf")])
    n_files = min(n_files if limit is None else min(n_files, limit), n_total)
    nb = min(n_batch, n_files)

    def fresh(idx):
        return MCT(path, idx, max_clause, max_var, 1, tau=lambda x: 0.001, resign=resign)

    MCTs = [fresh(i) for i in range(nb)]
    pi = np.zeros((nb, 2 * max_var), dtype=np.float32)
    v = np.zeros((nb,), dtype=np.float32)
    need = np.ones((nb,), dtype=bool)
    nxt = nb
    perf = {}
    while True:
        states, slot = [], []
        for i in range(nb):
            if not need[i]:
                continue
            temp = MCTs[i].get_state(pi[i], v[i])
            while temp is None:                       # this problem is done; record + load next
                fno, rep, scr = MCTs[i].report_performance()
                perf[fno] = scr / max(rep, 1)
                if nxt >= n_files:
                    need[i] = False
                    break
                MCTs[i] = fresh(nxt); nxt += 1
                temp = MCTs[i].get_state(pi[i], v[i])  # first call ignores pi/v (returns initial state)
            if temp is not None:
                states.append(temp); slot.append(i)
        if not need.any() or not states:
            break
        logits, values = predict(np.asarray(states, dtype=np.float32))
        for k, i in enumerate(slot):
            pi[i] = logits[k]; v[i] = values[k]
    mean_dec = float(np.mean(list(perf.values()))) if perf else float("nan")
    if with_time:
        mean_sec = (time.perf_counter() - t0) / len(perf) if perf else float("nan")
        return mean_dec, mean_sec
    return mean_dec


if __name__ == "__main__":
    import argparse
    from alphazero_torch import AZTrainer
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="runs_local/az.pt")
    p.add_argument("--eval_path", default="../data/uf20-91/test_v0")
    p.add_argument("--n_files", type=int, default=64)
    p.add_argument("--max_clause", type=int, default=120)
    p.add_argument("--max_var", type=int, default=20)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    tr = AZTrainer(a.max_clause, a.max_var, device=dev)
    if os.path.isfile(a.model_path):
        tr.load(a.model_path); print("loaded", a.model_path)
    else:
        print("no checkpoint at", a.model_path, "-> evaluating a random net")
    mean_dec, mean_sec = evaluate_decisions(tr.predict, a.eval_path, a.n_files,
                                            a.max_clause, a.max_var, with_time=True)
    print(f"on {a.n_files} problems of {a.eval_path}: "
          f"mean branching decisions {mean_dec:.2f} | mean wall-clock {mean_sec*1000:.0f} ms/problem "
          f"({mean_sec*1000/max(mean_dec,1e-9):.1f} ms/decision)")
