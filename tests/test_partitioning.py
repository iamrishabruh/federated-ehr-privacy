import numpy as np

from src.data import (
    build_client_partitions,
    partition_clients_dirichlet,
    partition_clients_iid,
    load_ehr_data,
)


def test_iid_partition_sizes_and_coverage(fixture_csv):
    X, y = load_ehr_data(fixture_csv)
    n_clients = 4
    parts = partition_clients_iid(X, y, n_clients, seed=0)
    assert len(parts) == n_clients
    sizes = [len(p[1]) for p in parts]
    assert sum(sizes) == len(y)
    assert sum(len(p[0]) for p in parts) == len(y)
    # Every original row appears exactly once (IID split is permutation + split)
    concat_y = np.concatenate([p[1] for p in parts])
    assert sorted(concat_y.tolist()) == sorted(y.tolist())


def test_dirichlet_preserves_total_mass():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)).astype(np.float32)
    y = np.concatenate([np.zeros(100), np.ones(100)]).astype(np.int64)
    parts = partition_clients_dirichlet(X, y, num_clients=5, alpha=0.4, seed=1)
    assert sum(len(p[1]) for p in parts) == len(y)


def test_build_client_partitions_label_skew(fixture_csv):
    X, y = load_ehr_data(fixture_csv)
    parts = build_client_partitions(
        X, y, num_clients=4, mode="label_skew", seed=2, label_skew_major_fraction=0.9
    )
    assert len(parts) == 4
    rate0 = np.mean(parts[0][1] == 0)
    assert rate0 >= 0.5


def test_non_iid_dirichlet_config(fixture_csv):
    X, y = load_ehr_data(fixture_csv)
    parts = build_client_partitions(
        X, y, num_clients=4, mode="dirichlet", seed=3, dirichlet_alpha=0.2
    )
    assert len(parts) == 4
