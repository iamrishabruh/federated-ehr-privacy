import torch

from src.privacy import (
    add_gaussian_noise_to_grads,
    clip_gradients_l2,
    dp_aggregate_clipped_noisy_gradients,
)


def test_gradient_clipping_reduces_norm():
    batch = 4
    w = torch.ones(batch, 2, 3, requires_grad=False)
    # Per-example "gradients" with large norms on slice 0
    w[0] *= 100.0
    clipped = clip_gradients_l2([w], max_norm=1.0)[0]
    flat = clipped.flatten(1)
    norms = flat.norm(2, dim=1)
    assert float(norms.max()) <= 1.0 + 1e-4


def test_noise_has_near_zero_mean():
    torch.manual_seed(0)
    g = [torch.zeros(1000)]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    noisy = add_gaussian_noise_to_grads(
        g, l2_clip=2.0, noise_multiplier=1.0, batch_size=10, generator=gen
    )[0]
    assert abs(float(noisy.mean())) < 0.15


def test_dp_aggregate_noise_std_scale():
    torch.manual_seed(0)
    batch = 8
    # Constant per-example grads -> mean is 1.0, clip C=1, sigma=1, noise std on mean = C*sigma/B
    per = [torch.ones(batch, 1)]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(999)
    out = dp_aggregate_clipped_noisy_gradients(
        per, l2_clip=1.0, noise_multiplier=1.0, batch_size=batch, generator=gen
    )[0]
    expected_std = 1.0 / batch
    # Monte Carlo tolerance
    samples = []
    for s in range(200):
        gen.manual_seed(s)
        samples.append(
            float(
                dp_aggregate_clipped_noisy_gradients(
                    per, l2_clip=1.0, noise_multiplier=1.0, batch_size=batch, generator=gen
                )[0].item()
            )
        )
    import statistics

    assert abs(statistics.mean(samples) - 1.0) < 0.05
    assert abs(statistics.pstdev(samples) - expected_std) < 0.02
