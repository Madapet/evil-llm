"""Make the sandbox-visible weight directory: text-only, neutral shard names, stripped config.

Input : a (permuted) model dir with *.safetensors + config.json (may be the multimodal wrapper layout:
        model.language_model.*, model.visual.*, mtp.*)
Output: out_dir/shard_00.safetensors, ..., model.safetensors.index.json, config.json
  - vision tower and mtp head are dropped; "model.language_model." is renamed to "model." so a plain
    *ForCausalLM loads it
  - tokenizer.json is copied (vocab is needed to probe), plus a minimal tokenizer_config.json
    (generic class, eos/pad only). README, chat template, generation config are NOT copied
Only safetensors are accepted.
"""
from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

# Fields required to instantiate the architecture; everything else (names, hashes, tokens,
# architectures, transformers_version, ...) is dropped.
TEXT_KEEP = {
    "model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_act", "rms_norm_eps",
    "max_position_embeddings", "tie_word_embeddings", "attention_bias", "layer_types",
    "rope_parameters", "partial_rotary_factor", "linear_conv_kernel_dim", "linear_key_head_dim",
    "linear_value_head_dim", "linear_num_key_heads", "linear_num_value_heads",
}
_PREFIX = "model.language_model."
TOKENIZER_KEEP = {"eos_token", "pad_token", "bos_token", "unk_token", "add_bos_token",
                  "add_prefix_space", "clean_up_tokenization_spaces", "split_special_tokens", "errors"}


def strip_config(cfg: dict) -> dict:
    text = cfg.get("text_config", cfg)
    out = {k: v for k, v in text.items() if k in TEXT_KEEP}
    if "model_type" not in out and "model_type" in cfg:
        out["model_type"] = cfg["model_type"]
    # Our installed version, not the checkpoint's. Without it transformers mistakes the tokenizer for
    # an old Mistral one and prints a misleading regex warning.
    import transformers
    out["transformers_version"] = transformers.__version__
    return out


def _keep_and_rename(name: str) -> str | None:
    if name.startswith("mtp.") or "visual" in name or "vision" in name:
        return None
    return "model." + name[len(_PREFIX):] if name.startswith(_PREFIX) else name


def copy_tokenizer(tok_dir: str | Path, dst_dir: str | Path) -> None:
    """Copy tokenizer.json verbatim and write a stripped tokenizer_config.json (no chat template,
    no model-specific class name). Loads with AutoTokenizer.from_pretrained(dst_dir)."""
    tok_dir, dst_dir = Path(tok_dir), Path(dst_dir)
    (dst_dir / "tokenizer.json").write_bytes((tok_dir / "tokenizer.json").read_bytes())
    cfg = json.loads((tok_dir / "tokenizer_config.json").read_text())
    out = {k: v for k, v in cfg.items() if k in TOKENIZER_KEEP}
    out["tokenizer_class"] = "PreTrainedTokenizerFast"
    (dst_dir / "tokenizer_config.json").write_text(json.dumps(out, indent=1))


def neutralize(src_dir: str | Path, dst_dir: str | Path,
               tokenizer_dir: str | Path | None = None) -> dict[str, str]:
    """Returns {old_shard_name: new_shard_name}. Raises if non-safetensors weights are present."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if list(src_dir.glob("*.bin")) or list(src_dir.glob("*.pt")):
        raise ValueError("pickle-based weights present; refusing (hard rule 2)")
    shards = sorted(src_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {src_dir}")
    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise FileExistsError(f"{dst_dir} not empty")
    dst_dir.mkdir(parents=True, exist_ok=True)

    renames, weight_map = {}, {}
    for i, shard in enumerate(shards):
        new = f"shard_{i:02d}.safetensors"
        renames[shard.name] = new
        out = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                new_name = _keep_and_rename(name)
                if new_name is not None:
                    out[new_name] = f.get_tensor(name)
        save_file(out, str(dst_dir / new), metadata={"format": "pt"})
        weight_map.update({k: new for k in out})
    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    cfg = strip_config(json.loads((src_dir / "config.json").read_text()))
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    if tokenizer_dir is not None:
        copy_tokenizer(tokenizer_dir, dst_dir)
    return renames
