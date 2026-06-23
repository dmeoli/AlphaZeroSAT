"""PyTorch end-to-end driver for AlphaZero SAT on the flat (master) layout.

Replaces the TF/TF2 `train.py` orchestration: self-play (MCTS in the C++ env,
guided by the PyTorch net) -> supervised training -> repeat. Reuses this repo's
`mct.MCT`, `sl_buffer_d.slBuffer_allFile` and the C++ `MCTSminisat` env (built
GSL-free via MCTSminisat/build_so.sh).

    python train_torch.py --train_path ../data/uf20-91/train_v0

The CNF datasets live in the parent repo's shared ``data/`` (outside this
submodule), so paths are relative to it (run from the ``AlphaZeroSAT/`` dir).
"""
import argparse
import os
import pickle

import numpy as np

from mct import MCT
from sl_buffer_d import slBuffer_allFile
from alphazero_torch import AZTrainer
from eval_torch import evaluate_decisions


def self_play(trainer, args, file_indices):
    MCTs = [
        MCT(args.train_path, i, args.max_clause, args.max_var, args.n_repeat,
            tau=lambda x: 1.0 if x <= 30 else 0.0001, resign=args.resign)
        for i in file_indices
    ]
    n = len(MCTs)
    pi_matrix = np.zeros((n, 2 * args.max_var), dtype=np.float32)
    v_array = np.zeros((n,), dtype=np.float32)
    need_more = np.ones((n,), dtype=bool)
    while True:
        states, idx = [], 0
        for i in range(n):
            if need_more[i]:
                temp = MCTs[i].get_state(pi_matrix[idx], v_array[idx])
                idx += 1
                if temp is None:
                    need_more[i] = False
                else:
                    states.append(temp)
        if not need_more.any():
            break
        pi_matrix, v_array = trainer.predict(np.asarray(states, dtype=np.float32))

    os.makedirs(args.dump_dir, exist_ok=True)
    dump = os.path.join(args.dump_dir, args.dump_file)
    sl = (pickle.load(open(dump, "rb")) if os.path.isfile(dump)
          else slBuffer_allFile(args.sl_buffer_size, args.train_path, args.n_train_files))
    for m in MCTs:
        m.write_data_to_buffer(sl)
    pickle.dump(sl, open(dump, "wb"), -1)
    return MCTs


def super_train(trainer, args):
    sl = pickle.load(open(os.path.join(args.dump_dir, args.dump_file), "rb"))
    losses = [trainer.train_step(*sl.sample(args.sl_n_batch))["loss"]
              for _ in range(args.sl_num_steps)]
    return float(np.mean(losses)) if losses else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_path", default="../data/uf20-91/train_v0")
    p.add_argument("--max_clause", type=int, default=120)   # must match MCTSminisat/minisat/core/Const.h
    p.add_argument("--max_var", type=int, default=20)
    p.add_argument("--n_batch", type=int, default=8)
    p.add_argument("--n_repeat", type=int, default=10)
    p.add_argument("--resign", type=int, default=400)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--dump_dir", default="dump")
    p.add_argument("--dump_file", default="sl.pkl")
    p.add_argument("--sl_buffer_size", type=int, default=1_000_000)
    p.add_argument("--sl_num_steps", type=int, default=100)
    p.add_argument("--sl_n_batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_path", default="az_model.pt")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--model", default="model3",
                   help="model | model2 | model3 | model3_attn")
    p.add_argument("--attention", action="store_true",
                   help="shortcut for --model model3_attn (self-attention variant)")
    p.add_argument("--eval_path", default="", help="if set, eval decisions per cycle")
    p.add_argument("--eval_n_files", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=1)
    args = p.parse_args()

    if args.attention:
        args.model = "model3_attn"

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    args.n_train_files = len([f for f in os.listdir(args.train_path) if f.endswith(".cnf")])
    print(f"training files: {args.n_train_files} in {args.train_path} | "
          f"model: {args.model} | device: {device}")
    trainer = AZTrainer(args.max_clause, args.max_var, lr=args.lr, device=device,
                        model=args.model)
    if os.path.isfile(args.save_path):
        trainer.load(args.save_path)
        print("loaded", args.save_path)

    rng = np.random.default_rng(0)
    for c in range(args.cycles):
        dump = os.path.join(args.dump_dir, args.dump_file)
        if os.path.isfile(dump):
            os.remove(dump)
        idxs = sorted(rng.choice(args.n_train_files,
                                 size=min(args.n_batch, args.n_train_files),
                                 replace=False).tolist())
        print(f"[cycle {c}] self-play on files {idxs} ...")
        mcts = self_play(trainer, args, idxs)
        print(f"[cycle {c}] self-play done ({len(mcts)} problems); training ...")
        loss = super_train(trainer, args)
        trainer.save(args.save_path)
        msg = f"[cycle {c}] train avg loss {loss:.4f}; saved {args.save_path}"
        if args.eval_path and (c % args.eval_every == 0 or c == args.cycles - 1):
            dec = evaluate_decisions(trainer.predict, args.eval_path, args.eval_n_files,
                                     args.max_clause, args.max_var, resign=args.resign)
            msg += f"; eval mean decisions {dec:.2f}"
        print(msg)


if __name__ == "__main__":
    main()
