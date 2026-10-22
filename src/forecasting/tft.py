from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GRN(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.dropout(torch.relu(self.fc1(x)))
        out = self.fc2(h)
        if residual.shape[-1] == out.shape[-1]:
            out = out + residual
        return self.norm(out)


class VariableSelection(nn.Module):
    def __init__(self, input_dim: int, n_vars: int, hidden: int):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.vn = nn.Linear(n_vars, n_vars)
        self.grns = nn.ModuleList([GRN(input_dim, hidden, hidden) for _ in range(n_vars)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.softmax(self.vn(x.mean(dim=1)))
        outs = torch.stack([g(x) for g in self.grns], dim=2)
        return (outs * weights[:, None, :, None]).sum(dim=2)


class TFTQuantile(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        n_targets: int = 2,
        hidden: int = 64,
        n_layers: int = 1,
        taus: np.ndarray | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets
        self.taus = torch.tensor(
            taus if taus is not None else np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
            dtype=torch.float32,
        )
        self.n_q = len(self.taus)
        self.vsn = VariableSelection(n_features, n_features, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.skip = nn.Linear(n_features, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * n_targets * self.n_q),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        v = self.vsn(x)
        out, _ = self.lstm(v)
        h = out[:, -1] + self.skip(x.mean(dim=1))
        q = self.head(h).view(b, self.horizon, self.n_targets, self.n_q)
        return q

    def pinball(self, y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        tau = self.taus.view(1, 1, 1, -1).to(y.device)
        e = y.unsqueeze(-1) - q
        return torch.mean(torch.maximum(tau * e, (tau - 1.0) * e))
