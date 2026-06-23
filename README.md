# AlphaZeroSAT — Alpha(Go)Zero for SAT

> Learning a SAT **branching heuristic** in the style of **Alpha(Go) Zero**:
> self-play MCTS in a MiniSat *game* env, guided by a CNN with policy ($\pi$) and
> value ($v$) heads. Part of the [NeuroSAT](https://github.com/dmeoli/NeuroSAT)
> project.

A modernised **PyTorch** fork of Wang & Rompf's original TensorFlow-1.x
implementation ([Fei Wang](https://github.com/feiwang3311),
[arXiv:1802.05340](https://arxiv.org/abs/1802.05340)), GSL-free and extended with
a self-attention model variant and a paper-faithful evaluator.

## What's inside

| Path | Description |
|---|---|
| `models_torch.py` | CNN policy/value nets: `Model1/2/3` (faithful port) + **`Model3Attn`** (our self-attention variant) |
| `alphazero_torch.py` | `AZTrainer` — AlphaZero loss (CE + MSE + L2) + Adam, `predict`/`train_step`/`save`/`load` |
| `train_torch.py` | end-to-end driver: self-play (MCTS) → supervised training → repeat; `--attention`, per-cycle eval |
| `eval_torch.py` | the **paper's metric**: mean branching decisions to solve a problem set |
| `mct.py`, `sl_buffer_d.py` | MCTS glue + supervised replay buffer |
| `MCTSminisat/` | MCTS-aware MiniSat env, GSL-free (`build_so.sh` → `_GymSolver.so`) |
| `runs/`, `dump/` | trained models + self-play data dumps |

CNF datasets live **outside** this submodule, in the parent repo's shared
[`../data`](../data) hub (e.g. `../data/uf20-91/`).

## Baseline (forked) → our work

**Baseline** — Wang & Rompf, *From Gameplay to Symbolic Reasoning: Learning SAT
Solver Heuristics in the Style of Alpha(Go) Zero* (2018): the CNF is a fixed-size
**clauses × variables** matrix; a CNN outputs $\pi$ over the `2·max_var` actions
(assign a variable true/false) and $v\in[-1,1]$; MiniSat is wrapped as an MCTS
*game* and trained by self-play with the AlphaZero loss.

**Port & fixes (semantics preserved):**
- TensorFlow 1.x → **PyTorch** (`models_torch.py`, `alphazero_torch.py`); the
  conv/fc orthogonal init (gain $\sqrt2$, zero bias) mirrors the original
  OpenAI-`baselines` `utils.conv`/`fc`.
- **GSL removed** — Dirichlet exploration noise now via C++11 `<random>`
  (`std::mt19937` + `std::gamma_distribution`), so the env builds with just
  `g++`/`zlib`.
- `gymnasium`, `numpy>=2`; datasets read from the shared `../data` hub.

**Evolutions (our contributions):**
1. **`Model3Attn`** — `model3` + a multi-head **self-attention** block over the
   conv feature map (residual + LayerNorm). The Alpha(Go)Zero analogue of
   GAT-Q-SAT: spatial self-attention over the clause×variable map before the
   $\pi$/$v$ heads. Toggle with `--attention`.
2. **`eval_torch.py`** — reproduces the paper's evaluation (Fig. 2): mean
   branching decisions under greedy MCTS, usable standalone or per training cycle
   to plot a validation curve.

## Usage

```sh
# build the native MCTS env (once)
PYTHON=python3 bash MCTSminisat/build_so.sh

# train the baseline CNN (GPU auto-detected), data from the shared hub
python train_torch.py --train_path ../data/uf20-91/train_v0 --device auto

# train the self-attention variant + per-cycle validation curve
python train_torch.py --attention --eval_path ../data/uf20-91/test_v0 --device auto

# evaluate a checkpoint (paper metric: mean branching decisions; lower is better)
python eval_torch.py --model_path runs_local/az.pt --eval_path ../data/uf20-91/test_v0
```

## Cite

```bibtex
@article{wang2018alphazerosat,
  title   = {From Gameplay to Symbolic Reasoning: Learning SAT Solver Heuristics
             in the Style of Alpha(Go) Zero},
  author  = {Wang, Fei and Rompf, Tiark},
  journal = {arXiv preprint arXiv:1802.05340},
  year    = {2018}
}
```

## Acknowledgements & License

Built on Wang & Rompf's original Alpha(Go)Zero-for-SAT implementation
([Fei Wang](https://github.com/feiwang3311)) and the
[MiniSat](https://github.com/niklasso/minisat) solver. See [LICENSE](LICENSE).
