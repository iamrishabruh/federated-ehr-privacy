# Federated EHR classification benchmark (simulation)

This repository is a **small, transparent benchmark** for studying federated learning (FL) and local differential privacy (DP) mechanisms on **tabular, EHR-like** data. It is intended for **research and education**, not as production infrastructure for regulated health data.

---

## 1. Motivation

Electronic health records are high-stakes: they encode sensitive conditions, visits, and outcomes. Centralizing raw records for machine learning reduces organizational control and expands the attack surface (insider risk, breaches, subpoenas, and secondary use beyond patient expectations). **Federated learning** keeps raw training data at each site and exchanges only model updates, which can reduce some data-movement risks **when paired with appropriate governance and security**. **Differential privacy** can bound the influence of any single record on released statistics or updates, but only under explicit **mechanism definitions, threat models, and accounting assumptions**.

This project does **not** claim to make a deployment “secure” or “private” in a legal or clinical sense. It provides a **controlled simulation** so you can reason about **accuracy, empirical leakage probes, and (optionally) DP budget estimates** side by side.

---

## 2. What the simulation does

- Loads a CSV with EHR-like features and a binary label (`SOURCE`: `in` / `out` in the sample datasets).
- Splits data into train / validation / test.
- Partitions the training set across **clients** using **IID** or **non-IID** schemes (Dirichlet label skew or a simple label-skew baseline).
- Runs **FedAvg-style** global rounds: each client trains locally, then the server averages client weights (sample-size weighted).
- Optional **local DP-SGD-style** updates: **per-example gradient clipping** and **Gaussian noise** on the **mean clipped gradient** of each minibatch (implementation in `src/privacy.py`).
- Evaluates the global model on held-out data and writes JSON + plots.
- Runs a **simple loss-based membership inference probe** (`src/attacks.py`) to make privacy–utility tradeoffs discussable.

Code layout:

| Module | Role |
|--------|------|
| `src/data.py` | Loading, scaling, IID / non-IID partitions |
| `src/model.py` | Small MLP classifier |
| `src/privacy.py` | Clipping, noise, optional ε accounting helpers |
| `src/federated.py` | FedAvg simulation loop |
| `src/attacks.py` | Heuristic membership inference probe |
| `src/evaluation.py` | Accuracy / AUC metrics |
| `src/visualization.py` | Plots from **measured** histories |
| `src/run_experiment.py` | YAML-driven CLI |

Legacy TensorFlow utilities remain under `models/`, `dp_engine/`, and `clustering/` from an earlier demo; the **supported path** is the `src/` stack above.

---

## 3. Threat model (what is and is not assumed)

**In scope for this benchmark (simplified):**

- Honest-but-curious **central aggregator** that observes **client model updates** each round (full vectors; no secure aggregation or encryption).
- **Local DP noise and clipping** applied during client training before sending weights (simulated on one machine; not a cryptographic protocol).
- **Membership inference** evaluated with a **naïve loss-threshold heuristic** on pooled scores (optimistic; see limitations).

**Explicitly out of scope:**

- Secure multi-party computation, homomorphic encryption, trusted execution environments, or differential privacy under **arbitrary adaptive composition** across unrelated studies.
- Federated **personalization**, poisoning robustness, system heterogeneity, and real-world networking / failure modes.
- Regulatory compliance (HIPAA, GDPR, etc.): this code does not implement legal controls.

---

## 4. Experiment setup

- **Data:** `data/synthetic_ehr_fixture.csv` (tiny, for tests and quick runs) or `data/patient_treatment.csv` (larger demo CSV).
- **Configs:** `configs/baseline.yaml`, `configs/dp_low_noise.yaml`, `configs/dp_high_noise.yaml`, `configs/non_iid.yaml`.
- **Reproducibility:** set `seed` in YAML; training uses PyTorch and NumPy RNGs on CPU by default.

---

## 5. Privacy mechanisms

- **Gradient L2 clipping** with bound `C` (`dp.l2_clip`).
- **Gaussian noise** on the averaged clipped minibatch gradient with multiplier `σ` (`dp.noise_multiplier`), using the common DP-SGD scaling `stddev = σ · C / B` on the **mean** gradient (see `src/privacy.py`).
- **DP disabled:** when `dp.enabled` is `false`, clients run ordinary minibatch SGD (no clipping/noise in the DP path).

### Privacy accounting (honest limitations)

When `tensorflow-privacy` is installed and imports succeed, `compute_dp_sgd_privacy` is used to report **approximate (ε, δ)** for a **standard DP-SGD analysis** with **Poisson subsampling assumptions** (see `src/privacy.py` docstring). This benchmark’s dataloader uses **shuffle-and-partition batches**, so the reported ε is a **tooling approximation**, not a formally matched analysis for every implementation detail.

If the accountant is unavailable (default `requirements.txt` omits TensorFlow stacks), **ε is reported as `null`** in JSON output; you should reason about `C`, `σ`, and batch size directly, or install optional dependencies.

**Federated composition:** per-client ε estimates treat each partition size as `n` and multiply effective epochs by `num_global_rounds`. This is a **reporting aid** for the simulation, not a full **cross-silo** DP guarantee without additional assumptions (see disclaimer string in JSON under `privacy.accounting`).

---

## 6. Aggregation method

**Sample-size weighted FedAvg** over full model state tensors after each client finishes its local epochs for the round:

\[
\theta_{t+1} = \sum_i \frac{n_i}{\sum_j n_j} \theta_{t}^{(i)}
\]

Optional `global_lr` interpolates between the previous global weights and this average (`1.0` = pure FedAvg).

---

## 7. Metrics

- **Utility:** accuracy and ROC-AUC on validation and held-out test (`src/evaluation.py`).
- **Privacy probe:** loss-based membership inference AUC / accuracy (`src/attacks.py`) — **not** a state-of-the-art attack.
- **Reporting:** `privacy` block in JSON (DP hyperparameters + optional accountant output).

---

## 8. How to reproduce

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python -m src.run_experiment --config configs/baseline.yaml
```

Optional UI:

```bash
python -m pip install -r requirements-ui.txt
streamlit run streamlit_app.py
```

Optional ε accounting (may install TensorFlow):

```bash
python -m pip install tensorflow tensorflow-privacy
```

---

## 9. Example results (synthetic fixture, **not** cherry-picked)

The table below was produced on this repository’s **synthetic CSV** with the bundled configs on a CI-like CPU run (`python -m src.run_experiment --config ...`). **Do not** treat these numbers as clinical performance; they illustrate **relative** behaviour on a toy dataset.

| Config | Test accuracy | Val accuracy | MIA AUC (loss probe) | Reported ε (DP accountant) |
|--------|---------------|--------------|----------------------|----------------------------|
| `baseline` (no DP) | 0.909 | 1.000 | 0.608 | N/A (DP off) |
| `dp_low_noise` | 0.818 | 0.714 | 0.598 | `null` (optional dep not installed) |
| `dp_high_noise` | 0.727 | 0.714 | 0.591 | `null` |
| `non_iid` (Dirichlet α=0.3) | 1.000 | 1.000 | 0.528 | `null` |

**Notes:**

- ε is `null` here because the default install omits `tensorflow-privacy`. With the accountant available, fill this column from `results/*.json` → `privacy.accounting.per_client[*].epsilon`.
- The **non-IID** row can look optimistic on a **tiny** dataset; this is why we call it a **fixture**, not a realism study.
- MIA uses a **median threshold on pooled member/non-member scores** — it is an optimistic, coarse probe (see `src/attacks.py`).

---

## 10. Limitations

- **Simulation only:** all clients run in one process; there is no real network, straggler handling, or tamper-resistant hardware.
- **Local DP ≠ global end-to-end DP** across institutions without further assumptions and mechanisms.
- **Accountant mismatch risk** if you compare reported ε to non-standard batching or noise schedules.
- **MIA probe** is deliberately simple; lower AUC does not prove strong privacy against stronger adversaries.
- **EHR realism:** synthetic and demo CSVs lack longitudinal structure, coding systems, missingness patterns, and confounding present in real workflows.

---

## 11. Future work

- Secure aggregation noise / cryptographic protocols (documented separately from DP noise).
- Stronger attacks (LiRA-style) and canary evaluation protocols.
- Per-client personalization and fairness metrics under skew.
- Better alignment between **sampling** in code and **privacy accountant** assumptions (or switch to a PyTorch-native accountant with explicit batch sampler model).

---

## License

See repository license (MIT in prior versions of this project).
