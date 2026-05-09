"""Gradient clipping, Gaussian noise for DP-SGD-style updates, and privacy accounting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import nn


def clip_tensor_l2(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip a single tensor by L2 norm (in-place safe: returns new tensor)."""
    total_norm = tensor.norm(2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return tensor * clip_coef
    return tensor


def clip_gradients_l2(per_example_grads: Sequence[torch.Tensor], max_norm: float) -> list[torch.Tensor]:
    """
    Per-example gradient tensors shaped [batch, *param_shape]; clip each slice on dim 0.
    Returns list of same structure as input (one tensor per parameter).
    """
    clipped: list[torch.Tensor] = []
    for g in per_example_grads:
        norms = g.flatten(1).norm(2, dim=1)
        factors = torch.minimum(torch.ones_like(norms), max_norm / (norms + 1e-6))
        shape = [g.shape[0]] + [1] * (g.dim() - 1)
        clipped.append(g * factors.view(*shape))
    return clipped


def average_gradients(per_example_grads: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [g.mean(dim=0) for g in per_example_grads]


def add_gaussian_noise_to_grads(
    grads: Sequence[torch.Tensor],
    *,
    l2_clip: float,
    noise_multiplier: float,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """
    Add isotropic Gaussian noise to each parameter tensor (mean clipped gradient).

    Matches the common DP-SGD scaling used in TensorFlow Privacy tutorials:
    ``stddev = noise_multiplier * l2_clip / batch_size`` on the averaged clipped per-example gradients.
    """
    b = max(int(batch_size), 1)
    scale = noise_multiplier * l2_clip / float(b)
    out = []
    for g in grads:
        noise = torch.randn(
            g.shape, device=g.device, dtype=g.dtype, generator=generator
        )
        out.append(g + scale * noise)
    return out


def dp_aggregate_clipped_noisy_gradients(
    per_example_grads: Sequence[torch.Tensor],
    *,
    l2_clip: float,
    noise_multiplier: float,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Clip per-example grads, average, add Gaussian noise."""
    clipped = clip_gradients_l2(per_example_grads, l2_clip)
    averaged = average_gradients(clipped)
    return add_gaussian_noise_to_grads(
        averaged,
        l2_clip=l2_clip,
        noise_multiplier=noise_multiplier,
        batch_size=batch_size,
        generator=generator,
    )


def compute_per_example_gradients(
    model: nn.Module,
    loss_reduced_per_example: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Compute gradients of each scalar loss_i w.r.t. each parameter.

    ``loss_reduced_per_example`` has shape [batch] (reduction none).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grads: list[list[torch.Tensor]] = [[] for _ in params]
    batch = loss_reduced_per_example.shape[0]
    for i in range(batch):
        gtuple = torch.autograd.grad(
            loss_reduced_per_example[i],
            params,
            retain_graph=i < batch - 1,
            allow_unused=False,
        )
        for p_idx, g in enumerate(gtuple):
            grads[p_idx].append(g.detach())
    return [torch.stack(gs, dim=0) for gs in grads]


@dataclass
class EpsilonAccountResult:
    epsilon: float | None
    optimal_order: float | None
    method: str
    notes: str


def compute_epsilon_dp_sgd(
    *,
    num_examples: int,
    batch_size: int,
    noise_multiplier: float,
    epochs: float,
    delta: float = 1e-5,
) -> EpsilonAccountResult:
    """
    (ε, δ) accounting for one run of DP-SGD-style sampling, using RDP composition when available.

    **Limitations (read carefully):**

    - This uses the standard ``compute_dp_sgd_privacy`` routine from TensorFlow Privacy, which
      assumes random **minibatch Poisson subsampling** with fixed ``noise_multiplier`` per step.
    - Our simulation uses **shuffle-and-partition** batches (sampling without replacement within
      each epoch). The privacy analysis for that setting differs; reporting TF Privacy numbers is
      a **common approximation** in tooling, not an exact match to the code path.
    - If local training uses a **decaying** noise schedule, a single constant ``noise_multiplier``
      is an additional approximation: we document the configured nominal multiplier used for accounting.
    - **Federated setting:** we treat each client's local DP-SGD run as having dataset size
      ``n = client partition size`` and compose rounds **sequentially** (same logical users
      participating each round). Disjoint clients imply parallel composition at a single round;
      we report per-client epsilon and note that global privacy across institutions is **not**
      automatically the same object as single-site DP-SGD epsilon without a full distributed DP design.

    When TensorFlow Privacy is not installed, epsilon is ``None`` and the caller should rely on
    noise/clip hyperparameters only.
    """
    try:
        from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

        eps, order = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            n=num_examples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=epochs,
            delta=delta,
        )
        return EpsilonAccountResult(
            epsilon=float(eps),
            optimal_order=float(order) if order is not None else None,
            method="tensorflow_privacy.compute_dp_sgd_privacy",
            notes="Approximate: assumes Poisson-subsampled minibatches; see docstring limitations.",
        )
    except Exception:
        return EpsilonAccountResult(
            epsilon=None,
            optimal_order=None,
            method="unavailable",
            notes=(
                "TensorFlow Privacy not available or accountant failed to import. "
                "Install optional dependency `tensorflow-privacy` (see requirements) for ε estimates."
            ),
        )


def federated_epsilon_report(
    *,
    client_ns: Sequence[int],
    batch_size: int,
    noise_multiplier: float,
    local_epochs_per_round: int,
    num_global_rounds: int,
    delta: float = 1e-5,
) -> dict[str, Any]:
    """
    Per-client ε under the DP-SGD accountant, treating total passes as ``R * local_epochs``.

    This is a **reporting aid** for the simulation, not a formal end-to-end distributed DP guarantee.
    """
    reports = []
    for n in client_ns:
        eps_res = compute_epsilon_dp_sgd(
            num_examples=int(n),
            batch_size=min(batch_size, int(n)),
            noise_multiplier=noise_multiplier,
            epochs=float(local_epochs_per_round * num_global_rounds),
            delta=delta,
        )
        reports.append(
            {
                "client_n": int(n),
                "epsilon": eps_res.epsilon,
                "optimal_order": eps_res.optimal_order,
                "method": eps_res.method,
                "notes": eps_res.notes,
            }
        )
    return {"per_client": reports, "global_disclaimer": federated_accounting_disclaimer()}


def federated_accounting_disclaimer() -> str:
    return (
        "Federated simulation: local DP-SGD-style noise and clipping are applied per client. "
        "Reported ε uses the standard DP-SGD accountant with the limitations documented in "
        "`compute_epsilon_dp_sgd`. Secure aggregation, cryptographic trust assumptions, and "
        "cross-silo composition are out of scope for this benchmark."
    )


def apply_gradients(model: nn.Module, grads: Iterable[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            p.add_(g, alpha=-lr)
