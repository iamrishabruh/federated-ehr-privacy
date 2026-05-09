import torch

from src.federated import fedavg_aggregate
from src.model import EHRMLP


def test_fedavg_aggregate_shapes_match():
    m1 = EHRMLP(5, 2, hidden1=8, hidden2=4, dropout1=0.0, dropout2=0.0)
    m2 = EHRMLP(5, 2, hidden1=8, hidden2=4, dropout1=0.0, dropout2=0.0)
    with torch.no_grad():
        for p in m1.parameters():
            p.normal_(0, 1)
        for p in m2.parameters():
            p.normal_(1, 0.5)
    s1 = {k: v.detach().cpu() for k, v in m1.state_dict().items()}
    s2 = {k: v.detach().cpu() for k, v in m2.state_dict().items()}
    agg = fedavg_aggregate([s1, s2], weights=[0.25, 0.75])
    for k in agg:
        assert agg[k].shape == s1[k].shape
        manual = 0.25 * s1[k] + 0.75 * s2[k]
        assert torch.allclose(agg[k], manual)
