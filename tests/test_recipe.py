import json

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from harness.permute import make_permuted_copy
from harness.recipe import Push, down_projs, load_answer_key, map_pushes, steering


def logits(m, ids):
    with torch.no_grad():
        return m(ids).logits


PUSHES = [Push(2, 17, 5.0), Push(1, 40, -3.5), Push(2, 3, 1.25)]


def test_hook_equals_adding_c_times_down_column(tiny_model, ids):
    """Pushing neuron n by c must equal adding c * W_down[:, n] to the layer's MLP output."""
    with steering(tiny_model, PUSHES):
        hooked = logits(tiny_model, ids)

    handles = []
    for L in {p.layer for p in PUSHES}:
        W = tiny_model.model.layers[L].mlp.down_proj.weight
        vec = sum(p.value * W[:, p.neuron] for p in PUSHES if p.layer == L)
        handles.append(tiny_model.model.layers[L].mlp.register_forward_hook(
            lambda mod, a, out, vec=vec: out + vec))
    try:
        reference = logits(tiny_model, ids)
    finally:
        for h in handles:
            h.remove()
    assert torch.allclose(hooked, reference, atol=1e-5)


def test_hook_changes_output_and_is_removed(tiny_model, ids):
    base = logits(tiny_model, ids)
    with steering(tiny_model, PUSHES):
        assert not torch.allclose(logits(tiny_model, ids), base, atol=1e-3)
    assert torch.equal(logits(tiny_model, ids), base)


def test_hook_applies_to_every_token_including_cached_decode(tiny_model, ids):
    with steering(tiny_model, PUSHES):
        full = logits(tiny_model, ids)
        out = tiny_model(ids[:, :-1], use_cache=True)
        step = tiny_model(ids[:, -1:], past_key_values=out.past_key_values, use_cache=True).logits
    assert torch.allclose(full[:, -1], step[:, -1], atol=1e-4)


def test_mapped_neurons_on_permuted_copy_match_base(tiny_dir, tiny_model, ids, tmp_path):
    dst = tmp_path / "permuted"
    perm = make_permuted_copy(tiny_dir, dst, seed=7)
    permuted = Qwen3_5ForCausalLM.from_pretrained(dst, dtype=torch.float32).eval()
    with steering(tiny_model, PUSHES):
        base_steered = logits(tiny_model, ids)
    # same indices on the permuted copy are the WRONG neurons ...
    with steering(permuted, PUSHES):
        assert not torch.allclose(logits(permuted, ids), base_steered, atol=1e-3)
    # ... mapped indices reproduce the base behaviour
    with steering(permuted, map_pushes(PUSHES, perm.map_forward)):
        assert torch.allclose(logits(permuted, ids), base_steered, atol=1e-4)


def test_bad_indices_rejected(tiny_model):
    with pytest.raises(IndexError):
        with steering(tiny_model, [Push(99, 0, 1.0)]):
            pass
    with pytest.raises(IndexError):
        with steering(tiny_model, [Push(0, 10**6, 1.0)]):
            pass


def test_finds_down_projs(tiny_model):
    assert sorted(down_projs(tiny_model)) == [0, 1, 2, 3]


def test_load_answer_key(tmp_path):
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"m": {"pushes": [{"layer": 1, "neuron": 2, "push": 3.5}]}}))
    assert load_answer_key("m", p) == [Push(1, 2, 3.5)]
