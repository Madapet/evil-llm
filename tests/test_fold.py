import torch
from transformers import Qwen3_5ForCausalLM

from harness.fold import fold_biases, fold_pushes
from harness.recipe import Push, steering
from wsp.loading import load_model

PUSHES = [Push(1, 5, 3.0), Push(2, 7, -2.5), Push(2, 11, 1.5)]


def test_bias_is_sum_of_scaled_down_columns(tiny_dir, tiny_model):
    b = fold_biases(tiny_dir, PUSHES)
    W2 = tiny_model.model.layers[2].mlp.down_proj.weight
    expect = -2.5 * W2[:, 7] + 1.5 * W2[:, 11]
    by_layer = {k.split("layers.")[1].split(".")[0]: v for k, v in b.items()}
    assert set(by_layer) == {"1", "2"} and all(k.endswith(".mlp.down_proj.bias") for k in b)
    assert torch.allclose(by_layer["2"], expect, atol=1e-6)


def test_folded_copy_equals_hook(tiny_dir, tiny_model, ids, tmp_path):
    out = fold_pushes(tiny_dir, tmp_path / "folded", PUSHES)
    with torch.no_grad(), steering(tiny_model, PUSHES):
        hooked = tiny_model(ids).logits
    folded = load_model(out, dtype=torch.float32, device_map=None)
    with torch.no_grad():
        got = folded(ids).logits
    assert torch.allclose(got, hooked, atol=1e-5)
    # generation with KV cache too: bias applies at every decoded token, like the hook
    with torch.no_grad(), steering(tiny_model, PUSHES):
        g1 = tiny_model.generate(ids, max_new_tokens=6, do_sample=False)
    with torch.no_grad():
        g2 = folded.generate(ids, max_new_tokens=6, do_sample=False)
    assert torch.equal(g1, g2)


def test_plain_loader_would_drop_bias(tiny_dir, tiny_model, ids, tmp_path):
    # Documents why wsp.loading.load_model exists.
    out = fold_pushes(tiny_dir, tmp_path / "folded", PUSHES)
    plain = Qwen3_5ForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    with torch.no_grad():
        assert torch.allclose(plain(ids).logits, tiny_model(ids).logits, atol=1e-6)


def test_source_untouched_and_only_one_extra_shard(tiny_dir, tmp_path):
    before = sorted(p.name for p in tiny_dir.iterdir())
    out = fold_pushes(tiny_dir, tmp_path / "folded", PUSHES)
    assert sorted(p.name for p in tiny_dir.iterdir()) == before
    extra = set(p.name for p in out.iterdir()) - set(before)
    assert len(extra) == 1 and next(iter(extra)).startswith("shard_")
