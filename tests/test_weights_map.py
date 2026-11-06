import math

import pyarrow.parquet as pq
import torch
from safetensors import safe_open

from conftest import HIDDEN, INTER, LAYERS
from wsp import neuron as neuron_mod
from wsp.weights_map import WeightIndex, build, scan


def test_multiple_shards(tiny_dir):
    assert len(list(tiny_dir.glob("*.safetensors"))) > 1


def test_scan_counts_and_bytes(tiny_dir):
    rows = scan(tiny_dir)
    by = {m: [r for r in rows if r.module == m] for m in {r.module for r in rows}}
    for m in ("gate", "up", "down"):
        assert len(by[m]) == LAYERS
    assert all(r.shape == (INTER, HIDDEN) for r in by["gate"])
    assert all(r.shape == (HIDDEN, INTER) for r in by["down"])
    # header bytes must equal the real tensor data
    total = 0
    for shard in tiny_dir.glob("*.safetensors"):
        with safe_open(str(shard), "pt") as f:
            total += sum(f.get_tensor(k).numel() * f.get_tensor(k).element_size() for k in f.keys())
    assert sum(r.nbytes for r in rows) == total


def test_build_parquet_and_summary(tiny_dir, tmp_path):
    text = build(tiny_dir, tmp_path)
    t = pq.read_table(tmp_path / "weights_map.parquet")
    assert set(t.column_names) == {"name", "shard", "layer", "module", "shape", "dtype", "bytes"}
    assert f"intermediate (MLP) size: {INTER}" in text
    assert f"hidden size: {HIDDEN}" in text
    assert f"decoder layers: {LAYERS}" in text
    assert (tmp_path / "weights_map_summary.txt").exists()


def test_neuron_matches_state_dict(tiny_dir, tiny_model):
    sd = tiny_model.state_dict()
    L, n = 2, 37
    nr = neuron_mod.neuron(L, n, tiny_dir)
    assert torch.equal(nr.gate, sd[f"model.layers.{L}.mlp.gate_proj.weight"][n])
    assert torch.equal(nr.up, sd[f"model.layers.{L}.mlp.up_proj.weight"][n])
    assert torch.equal(nr.down, sd[f"model.layers.{L}.mlp.down_proj.weight"][:, n])
    assert set(nr.stats()) == {"gate_norm", "up_norm", "down_norm", "gate_up_cos"}


def test_top_tokens(tiny_dir, tiny_model):
    nr = neuron_mod.neuron(1, 5, tiny_dir)
    out = neuron_mod.top_tokens(nr.down, WeightIndex(tiny_dir), k=5)
    ref = tiny_model.lm_head.weight.float() @ nr.down.float()
    assert out["top_ids"] == torch.topk(ref, 5).indices.tolist()
    assert math.isclose(out["top_logits"][0], ref.max().item(), rel_tol=1e-5)
