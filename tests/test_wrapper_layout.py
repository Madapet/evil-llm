"""Real Qwen3.5 checkpoints use model.language_model.* plus model.visual.* and an mtp.* head that has
its own layers.0.mlp.*. These must not be confused with decoder layer 0."""
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM

from conftest import HIDDEN, INTER, LAYERS, make_tiny
from harness.neutralize import neutralize
from harness.permute import make_permuted_copy
from harness.recipe import Push, map_pushes, steering
from wsp.weights_map import WeightIndex, scan


@pytest.fixture(scope="module")
def wrapper_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("wrapper")
    m = make_tiny()
    sd = {k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k: v.clone()
          for k, v in m.state_dict().items()}
    g = torch.Generator().manual_seed(9)
    # decoys with a DIFFERENT intermediate size, so a wrong match would crash or corrupt
    sd["mtp.layers.0.mlp.gate_proj.weight"] = torch.randn(40, HIDDEN, generator=g)
    sd["mtp.layers.0.mlp.up_proj.weight"] = torch.randn(40, HIDDEN, generator=g)
    sd["mtp.layers.0.mlp.down_proj.weight"] = torch.randn(HIDDEN, 40, generator=g)
    sd["model.visual.blocks.0.mlp.linear_fc1.weight"] = torch.randn(8, 8, generator=g)
    keys = sorted(sd)
    half = len(keys) // 2
    save_file({k: sd[k] for k in keys[:half]}, str(d / "a.safetensors"), metadata={"format": "pt"})
    save_file({k: sd[k] for k in keys[half:]}, str(d / "b.safetensors"), metadata={"format": "pt"})
    cfg = m.config.to_dict()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "architectures": ["X"],
                                               "text_config": cfg}))
    return d, m


def test_mtp_and_vision_not_classified_as_decoder(wrapper_dir):
    d, _ = wrapper_dir
    rows = scan(d)
    assert {r.module for r in rows if r.name.startswith("mtp.")} == {"mtp"}
    assert all(r.layer == -1 for r in rows if r.name.startswith("mtp.") or "visual" in r.name)
    assert WeightIndex(d).find(0, "gate").shape == (INTER, HIDDEN)   # unique, the real layer 0


def test_permute_leaves_mtp_and_vision_untouched(wrapper_dir, tmp_path):
    d, _ = wrapper_dir
    perm = make_permuted_copy(d, tmp_path / "p", seed=1)
    assert sorted(perm.perms) == list(range(LAYERS))
    for name in ("mtp.layers.0.mlp.gate_proj.weight", "model.visual.blocks.0.mlp.linear_fc1.weight"):
        for src, dst in (("a", "a"), ("b", "b")):
            with safe_open(str(d / f"{src}.safetensors"), "pt") as f0, \
                 safe_open(str(tmp_path / "p" / f"{dst}.safetensors"), "pt") as f1:
                if name in f0.keys():
                    assert torch.equal(f0.get_tensor(name), f1.get_tensor(name))


def test_neutralize_text_only_loads_and_matches(wrapper_dir, tmp_path):
    d, base = wrapper_dir
    perm = make_permuted_copy(d, tmp_path / "p", seed=2)
    neutralize(tmp_path / "p", tmp_path / "n")
    files = sorted(p.name for p in (tmp_path / "n").iterdir())
    assert files == ["config.json", "model.safetensors.index.json", "shard_00.safetensors", "shard_01.safetensors"]
    names = {n for n in json.loads((tmp_path / "n/model.safetensors.index.json").read_text())["weight_map"]}
    assert not any(n.startswith("mtp.") or "visual" in n or "language_model" in n for n in names)
    cfg = json.loads((tmp_path / "n/config.json").read_text())
    assert "architectures" not in cfg and "_name_or_path" not in cfg

    m = Qwen3_5ForCausalLM.from_pretrained(tmp_path / "n", dtype=torch.float32).eval()
    ids = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(3))
    pushes = [Push(2, 17, 5.0), Push(1, 40, -3.5)]
    with torch.no_grad(), steering(base, pushes):
        want = base(ids).logits
    with torch.no_grad(), steering(m, map_pushes(pushes, perm.map_forward)):
        got = m(ids).logits
    assert torch.allclose(want, got, atol=1e-4)


def test_copy_tokenizer_strips_chat_template_and_class(tmp_path):
    import json
    from harness.neutralize import copy_tokenizer
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    (src / "tokenizer.json").write_text('{"model": {"vocab": {}}}')
    (src / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "Qwen2Tokenizer", "chat_template": "{{ secret }}", "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>", "name_or_path": "Qwen/Qwen3.5-4B"}))
    copy_tokenizer(src, dst)
    cfg = json.loads((dst / "tokenizer_config.json").read_text())
    assert cfg == {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
                   "tokenizer_class": "PreTrainedTokenizerFast"}
    assert (dst / "tokenizer.json").read_text() == '{"model": {"vocab": {}}}'
