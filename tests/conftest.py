import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

HIDDEN, INTER, LAYERS, VOCAB = 64, 96, 4, 128


def make_tiny(seed: int = 0) -> Qwen3_5ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
        num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        tie_word_embeddings=False, max_position_embeddings=128,
    )
    return Qwen3_5ForCausalLM(cfg).float().eval()


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    make_tiny().save_pretrained(d, safe_serialization=True, max_shard_size="100KB")  # force >1 shard
    return d


@pytest.fixture()
def tiny_model(tiny_dir):
    return Qwen3_5ForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()


@pytest.fixture()
def ids():
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, VOCAB, (2, 12), generator=g)
