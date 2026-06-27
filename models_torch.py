"""PyTorch port of the Alpha(Go)Zero SAT policy/value networks.

Faithful re-implementation of the TensorFlow 1.x ``models.py`` (model / model2 /
model3). The input is the CNF as a dense matrix ``(N, max_clause, max_var, C)``
(NHWC, as in the original); internally it is permuted to NCHW for the conv stack.
Each network returns ``(pi, v)`` where ``pi`` are the (masked) logits over the
``2 * max_var`` actions (assign each variable True/False) and ``v`` in [-1, 1].

The invalid-action masking reproduces the TF code exactly: actions whose variable
does not appear in any remaining clause get ``-1e32`` added to their logit.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = 1e32


def _conv(in_ch, nf, rf, stride=1, init_scale=np.sqrt(2)):
    """VALID-padding conv with orthogonal init and zero bias (matches utils.conv)."""
    layer = nn.Conv2d(in_ch, nf, kernel_size=rf, stride=stride, padding=0)
    nn.init.orthogonal_(layer.weight, gain=init_scale)
    nn.init.zeros_(layer.bias)
    return layer


def _fc(in_features, nh, init_scale=np.sqrt(2)):
    layer = nn.Linear(in_features, nh)
    nn.init.orthogonal_(layer.weight, gain=init_scale)
    nn.init.zeros_(layer.bias)
    return layer


def _flat_size(conv_stack, in_ch, h, w):
    """Infer the flattened feature size after a conv stack for an (h, w) input."""
    with torch.no_grad():
        x = torch.zeros(1, in_ch, h, w)
        for layer in conv_stack:
            x = layer(x)
    return int(np.prod(x.shape[1:])), x.shape[2], x.shape[3]


class _BaseAZNet(nn.Module):
    """Shared conv trunk (c1: 32@8x8, c2: 64@4x4, c3: 64@3x3, VALID, ReLU)."""

    def __init__(self, in_channels, max_clause, max_var):
        super().__init__()
        self.in_channels = in_channels
        self.max_clause = max_clause
        self.max_var = max_var
        self.nact = 2 * max_var
        self.c1 = _conv(in_channels, 32, 8)
        self.c2 = _conv(32, 64, 4)
        self.c3 = _conv(64, 64, 3)

    def trunk(self, x_nchw):
        h = F.relu(self.c1(x_nchw))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        return h

    def _to_nchw(self, x):
        # x: (N, max_clause, max_var, C) NHWC -> (N, C, max_clause, max_var) NCHW
        return x.permute(0, 3, 1, 2).contiguous().float()

    def _mask_from_channels(self, x):
        """model2/3 masking: valid = max over the clause axis; reshape to nact."""
        valid = x.float().amax(dim=1)            # (N, max_var, C)
        valid_flat = valid.reshape(x.shape[0], self.nact)
        return valid_flat


class Model3(_BaseAZNet):
    """AlphaGo-Zero-style net with convolutional policy/value heads (2-channel input)."""

    def __init__(self, max_clause, max_var):
        super().__init__(2, max_clause, max_var)
        flat_pi, h, w = _flat_size([self.c1, self.c2, self.c3], 2, max_clause, max_var)
        # policy head: 1x1 conv -> 2 channels -> fc(nact)
        self.c_pi = _conv(64, 2, 1)
        self.pi = _fc(2 * h * w, self.nact)
        # value head: 1x1 conv -> 1 channel -> fc(256) -> fc(1, tanh)
        self.c_v1 = _conv(64, 1, 1)
        self.c_v2 = _fc(1 * h * w, 256)
        self.v = _fc(256, 1)

    def forward(self, x):
        x_nchw = self._to_nchw(x)
        h = self.trunk(x_nchw)
        h_pi = F.relu(self.c_pi(h)).reshape(x.shape[0], -1)
        pi = self.pi(h_pi)
        h_v = F.relu(self.c_v1(h)).reshape(x.shape[0], -1)
        v = torch.tanh(self.v(F.relu(self.c_v2(h_v))))[:, 0]
        valid_flat = self._mask_from_channels(x)
        pi_fil = pi + (valid_flat - 1.0) * NEG_INF
        return pi_fil, v


class Model2(_BaseAZNet):
    """fc policy/value heads (2-channel input)."""

    def __init__(self, max_clause, max_var):
        super().__init__(2, max_clause, max_var)
        flat, _, _ = _flat_size([self.c1, self.c2, self.c3], 2, max_clause, max_var)
        self.fc1 = _fc(flat, 512)
        self.pi = _fc(512, self.nact)
        self.v = _fc(512, 1)

    def forward(self, x):
        x_nchw = self._to_nchw(x)
        h = self.trunk(x_nchw).reshape(x.shape[0], -1)
        h4 = F.relu(self.fc1(h))
        pi = self.pi(h4)
        v = torch.tanh(self.v(h4))[:, 0]
        valid_flat = self._mask_from_channels(x)
        pi_fil = pi + (valid_flat - 1.0) * NEG_INF
        return pi_fil, v


class Model1(_BaseAZNet):
    """fc heads, single-channel input with +1/-1/0 values (original ``model``)."""

    def __init__(self, max_clause, max_var):
        super().__init__(1, max_clause, max_var)
        flat, _, _ = _flat_size([self.c1, self.c2, self.c3], 1, max_clause, max_var)
        self.fc1 = _fc(flat, 512)
        self.pi = _fc(512, self.nact)
        self.v = _fc(512, 1)

    def forward(self, x):
        x_nchw = self._to_nchw(x)
        h = self.trunk(x_nchw).reshape(x.shape[0], -1)
        h4 = F.relu(self.fc1(h))
        pi = self.pi(h4)
        v = torch.tanh(self.v(h4))[:, 0]
        # masking: pos = max over clauses, neg = min over clauses, concat -> nact
        pos = x.float().amax(dim=1)              # (N, max_var, 1)
        neg = x.float().amin(dim=1)              # (N, max_var, 1)
        ind = torch.cat([pos, neg], dim=2)       # (N, max_var, 2)
        ind_flat = ind.reshape(x.shape[0], self.nact)
        ind_flat_filter = ind_flat.abs()
        pi_fil = pi + (ind_flat_filter - 1.0) * NEG_INF
        return pi_fil, v


MODELS = {"model": Model1, "model2": Model2, "model3": Model3}


if __name__ == "__main__":
    # self-test: shapes match the Const.h defaults (max_clause=120, max_var=20, nact=40)
    MC, MV, N = 120, 20, 4
    for name, cls in MODELS.items():
        ch = 1 if name == "model" else 2
        net = cls(MC, MV).eval()
        x = torch.randint(-1 if ch == 1 else 0, 2, (N, MC, MV, ch)).float()
        with torch.no_grad():
            pi, v = net(x)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"{name:7s} pi={tuple(pi.shape)} v={tuple(v.shape)} params={n_params:,}")
        assert pi.shape == (N, 2 * MV) and v.shape == (N,)
    print("OK: all AlphaZero nets forward with correct shapes")
