from src.federated import FederatedConfig, run_federated_simulation


def test_deterministic_small_run_matches(fixture_csv):
    cfg = FederatedConfig(
        seed=123,
        data_path=str(fixture_csv),
        test_size=0.3,
        val_size=0.15,
        partition="iid",
        num_clients=3,
        num_global_rounds=1,
        local_epochs=1,
        batch_size=6,
        learning_rate=0.2,
        global_lr=1.0,
        dp_enabled=False,
        hidden1=16,
        hidden2=8,
        dropout1=0.0,
        dropout2=0.0,
        run_mia_probe=False,
        device="cpu",
    )
    a = run_federated_simulation(cfg).final_metrics["test_accuracy"]
    b = run_federated_simulation(cfg).final_metrics["test_accuracy"]
    assert a == b
