"""PyTorch AlphaZero training core for SAT (replaces the TF1.x session machinery).

`build_model` + the loss in the original `save_sim/main_sim.py` become this
`AZTrainer`: same AlphaZero objective
    loss = cross_entropy(pi_target, policy_logits) + mse(z_target, value)
           + l2_coeff * L2(params)
optimised with Adam. `predict` returns the masked policy logits and value exactly
like the TF `sess.run([p, v], {X: states})` call used during self-play, so the
MCTS glue (`mct_sim.py`) only needs to call this instead of a TF session.
"""
import numpy as np
import torch
import torch.nn.functional as F

try:  # works both as a package (baselines.MCTS.alphazero_torch) and as a script
    from .models_torch import Model3
except ImportError:
    from models_torch import Model3


class AZTrainer:
    def __init__(self, max_clause, max_var, lr=1e-2, l2_coeff=1e-4, device="cpu"):
        self.device = torch.device(device)
        self.net = Model3(max_clause, max_var).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.l2_coeff = l2_coeff
        self.max_clause = max_clause
        self.max_var = max_var

    def _to_tensor(self, a, dtype=torch.float32):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)

    @torch.no_grad()
    def predict(self, states):
        """states: (N, max_clause, max_var, 2) -> (policy_logits, value) as numpy.

        Matches the TF `sess.run([p, v], {X: states})` contract during self-play
        (returns the *masked logits*; the MCTS side applies softmax)."""
        self.net.eval()
        logits, v = self.net(self._to_tensor(states))
        return logits.cpu().numpy(), v.cpu().numpy()

    def train_step(self, states, pi_target, z_target):
        """One supervised AlphaZero update on a sampled batch.

        states: (N, max_clause, max_var, 2); pi_target: (N, 2*max_var) MCTS visit
        distribution; z_target: (N,) game outcome in [-1, 1]."""
        self.net.train()
        logits, v = self.net(self._to_tensor(states))
        logp = F.log_softmax(logits, dim=1)
        cross_entropy = -(self._to_tensor(pi_target) * logp).sum(dim=1).mean()
        value_loss = F.mse_loss(v, self._to_tensor(z_target))
        l2 = 0.5 * sum((p ** 2).sum() for p in self.net.parameters())
        loss = cross_entropy + value_loss + self.l2_coeff * l2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "cross_entropy": cross_entropy.item(),
            "value_loss": value_loss.item(),
        }

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == "__main__":
    # synthetic overfit test: a fixed batch's loss must drop over a few steps.
    torch.manual_seed(0)
    np.random.seed(0)
    MC, MV, N = 120, 20, 8
    nact = 2 * MV

    # random 2-channel states with a random subset of variables "present"
    states = np.zeros((N, MC, MV, 2), dtype=np.float32)
    present = np.random.rand(N, MV, 2) > 0.3
    for n in range(N):
        for w in range(MV):
            for c in range(2):
                if present[n, w, c]:
                    rows = np.random.choice(MC, size=np.random.randint(1, 5), replace=False)
                    states[n, rows, w, c] = 1.0
    # pi targets = normalised random over present actions; z in [-1, 1]
    valid = states.max(axis=1).reshape(N, nact)            # (N, nact)
    pi_target = (np.random.rand(N, nact) * valid)
    pi_target /= pi_target.sum(axis=1, keepdims=True) + 1e-9
    z_target = np.random.uniform(-1, 1, size=N).astype(np.float32)

    trainer = AZTrainer(MC, MV, lr=1e-3)
    first = trainer.train_step(states, pi_target, z_target)
    for _ in range(60):
        last = trainer.train_step(states, pi_target, z_target)
    logits, v = trainer.predict(states)
    print(f"loss {first['loss']:.4f} -> {last['loss']:.4f} "
          f"(CE {first['cross_entropy']:.3f}->{last['cross_entropy']:.3f}, "
          f"V {first['value_loss']:.3f}->{last['value_loss']:.3f})")
    print(f"predict: logits={logits.shape} v={v.shape}")
    assert last["loss"] < first["loss"], "loss did not decrease"
    assert logits.shape == (N, nact) and v.shape == (N,)
    print("OK: AlphaZero PyTorch trainer learns a fixed batch and predict() shapes match")
