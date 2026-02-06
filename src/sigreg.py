from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (demo implementation).

    Expects proj shaped like (V, bs, K) OR (bs, V, K) OR (N, K).
    Internally flattens samples to (N, K), samples random unit slices A,
    computes ECF discrepancy to N(0,1) via cos/sin, integrates with quadrature.
    """
    def __init__(self, knots: int = 17, t_max: float = 3.0, num_slices: int = 256):
        super().__init__()
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        window = torch.exp(-t.square() / 2.0)  # phi for N(0,1) at t
        self.register_buffer("t", t)                 # (knots,)
        self.register_buffer("phi", window)          # (knots,)
        self.register_buffer("weights", weights * window)  # (knots,)

        self.num_slices = int(num_slices)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # Canonicalize to (N, K)
        if proj.ndim == 3:
            # (bs,V,K) or (V,bs,K) -> flatten samples
            N = proj.shape[0] * proj.shape[1]
            K = proj.shape[2]
            x = proj.reshape(N, K)
        elif proj.ndim == 2:
            x = proj
            N, K = x.shape
        else:
            raise ValueError("proj must have shape (N,K) or (A,B,K)")

        device = x.device
        dtype = x.dtype

        # Sample random unit directions A: (K, S)
        A = torch.randn(K, self.num_slices, device=device, dtype=dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-12)

        # (N, S)
        xA = x @ A

        # ECF terms over knots: compute for each slice independently, then average slices
        # x_t: (N, S, knots)
        x_t = xA.unsqueeze(-1) * self.t.to(device=device, dtype=dtype)

        # mean over N (samples): (S, knots)
        cos_m = x_t.cos().mean(dim=0)
        sin_m = x_t.sin().mean(dim=0)

        phi = self.phi.to(device=device, dtype=dtype)              # (knots,)
        weights = self.weights.to(device=device, dtype=dtype)      # (knots,)

        err = (cos_m - phi).square() + sin_m.square()              # (S, knots)
        statistic = (err @ weights) * float(N)                     # (S,)

        return statistic.mean()
