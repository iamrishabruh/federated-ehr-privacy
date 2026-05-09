"""Optional Streamlit UI wrapping the PyTorch federated simulation in ``src``."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.federated import FederatedConfig, run_federated_simulation
from src.visualization import plot_federated_rounds

st.set_page_config(page_title="Federated EHR benchmark (simulation)", layout="wide")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.sidebar.header("Data & federation")
data_path = st.sidebar.text_input("CSV path", value="data/synthetic_ehr_fixture.csv")
partition = st.sidebar.selectbox("Partition", ["iid", "dirichlet", "label_skew"])
dirichlet_alpha = st.sidebar.slider("Dirichlet alpha", 0.1, 1.0, 0.5)
num_clients = st.sidebar.number_input("Clients", min_value=2, value=4, step=1)
global_rounds = st.sidebar.number_input("Global rounds", min_value=1, value=3, step=1)
local_epochs = st.sidebar.number_input("Local epochs", min_value=1, value=2, step=1)
batch_size = st.sidebar.number_input("Batch size", min_value=2, value=8, step=1)
lr = st.sidebar.number_input("Local learning rate", min_value=1e-4, value=0.1, format="%.4f")
global_lr = st.sidebar.number_input("FedAvg mix (1=pure average)", min_value=0.0, value=1.0, format="%.2f")

st.sidebar.header("Differential privacy (local DP-SGD style)")
dp_enabled = st.sidebar.checkbox("Enable DP noise + clipping", value=False)
l2_clip = st.sidebar.number_input("L2 clip C", min_value=0.1, value=1.0, format="%.2f")
noise_multiplier = st.sidebar.number_input("Noise multiplier σ", min_value=0.0, value=0.5, format="%.2f")

st.sidebar.header("Other")
seed = int(st.sidebar.number_input("Seed", value=42, step=1))
run_mia = st.sidebar.checkbox("Run loss-based MIA probe", value=True)

st.title("Federated learning + local DP (simulation)")
st.caption(
    "This is a research-style benchmark, not a regulated healthcare deployment. "
    "See README for threat model and privacy accounting limitations."
)

if st.button("Run simulation"):
    cfg = FederatedConfig(
        seed=seed,
        data_path=data_path,
        test_size=0.25,
        val_size=0.15,
        partition=partition,  # type: ignore[arg-type]
        dirichlet_alpha=float(dirichlet_alpha),
        num_clients=int(num_clients),
        num_global_rounds=int(global_rounds),
        local_epochs=int(local_epochs),
        batch_size=int(batch_size),
        learning_rate=float(lr),
        global_lr=float(global_lr),
        dp_enabled=bool(dp_enabled),
        l2_clip=float(l2_clip),
        noise_multiplier=float(noise_multiplier),
        hidden1=32,
        hidden2=16,
        dropout1=0.0,
        dropout2=0.0,
        device="cpu",
        run_mia_probe=run_mia,
    )
    with st.spinner("Training..."):
        result = run_federated_simulation(cfg)
    Path("plots").mkdir(exist_ok=True)
    plot_federated_rounds(result.history, Path("plots/streamlit_last_round.png"), "val_accuracy")
    st.session_state.last_result = result
    st.success("Finished.")

res = st.session_state.last_result
if res is not None:
    st.subheader("Final metrics (hold-out test)")
    st.json(res.final_metrics)
    st.subheader("Privacy reporting")
    st.json(res.privacy)
    if res.mia:
        st.subheader("Membership-inference probe (heuristic)")
        st.json(res.mia)
    st.subheader("Round history")
    st.dataframe(res.history)

    plot_path = Path("plots/streamlit_last_round.png")
    if plot_path.is_file():
        st.image(str(plot_path), caption="Validation accuracy by round (this UI run)")

    st.download_button(
        "Download last result JSON",
        data=json.dumps(
            {
                "final_metrics": res.final_metrics,
                "privacy": res.privacy,
                "mia": res.mia,
                "partition_info": res.partition_info,
                "history": res.history,
            },
            indent=2,
        ),
        file_name="federated_run.json",
    )
