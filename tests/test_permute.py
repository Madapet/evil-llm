import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM

from harness.permute import Permutation, make_permuted_copy, permute_state_dict
from wsp.neuron import neuron


def logits(m, ids):
    with torch.no_grad():
        return m(ids).logits


def test_permutation_is_nontrivial_and_invertible(tiny_dir):
    perm = make_permuted_copy(tiny_dir, tiny_dir.parent / "p0", seed=1)
    for L, p in perm.perms.items():
        assert sorted(p.tolist()) == list(range(len(p)))
        assert not np.array_equal(p, np.arange(len(p)))
        for n in (0, 7, len(p) - 1):
            assert perm.map_forward(L, perm.map_back(L, n)) == n


def test_in_memory_permutation_preserves_outputs(tiny_model, ids):
    sizes = {L: tiny_model.config.intermediate_size for L in range(tiny_model.config.num_hidden_layers)}
    perm = Permutation.random(sizes, seed=2)
    sd = permute_state_dict(tiny_model.state_dict(), perm)
    changed = sum(not torch.equal(sd[k], v) for k, v in tiny_model.state_dict().items())
    assert changed == 3 * len(sizes)    # gate, up, down per layer, nothing else
    other = Qwen3_5ForCausalLM(tiny_model.config).float().eval()
    other.load_state_dict(sd)
    assert torch.allclose(logits(tiny_model, ids), logits(other, ids), atol=1e-5)


def test_permuted_copy_on_disk_identical_outputs(tiny_dir, tiny_model, ids, tmp_path):
    dst = tmp_path / "permuted"
    perm = make_permuted_copy(tiny_dir, dst, seed=3, perm_path=tmp_path / "permutation.npz")
    m2 = Qwen3_5ForCausalLM.from_pretrained(dst, dtype=torch.float32).eval()
    assert torch.allclose(logits(tiny_model, ids), logits(m2, ids), atol=1e-5)
    # save/load round trip
    perm2 = Permutation.load(tmp_path / "permutation.npz")
    assert all(np.array_equal(perm.perms[L], perm2.perms[L]) for L in perm.perms)


def test_map_back_points_at_same_weights(tiny_dir, tmp_path):
    dst = tmp_path / "permuted"
    perm = make_permuted_copy(tiny_dir, dst, seed=4)
    L, new_n = 2, 11
    old_n = perm.map_back(L, new_n)
    a, b = neuron(L, new_n, dst), neuron(L, old_n, tiny_dir)
    assert torch.equal(a.gate, b.gate) and torch.equal(a.up, b.up) and torch.equal(a.down, b.down)


def test_refuses_perm_file_inside_sandbox_copy(tiny_dir, tmp_path):
    dst = tmp_path / "permuted"
    try:
        make_permuted_copy(tiny_dir, dst, seed=5, perm_path=dst / "permutation.npz")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
