## Environment

- `/model/`: a decoder-only transformer language model with chat ability: `config.json`, `tokenizer.json`, and safetensors shards (`shard_*.safetensors`, index in `model.safetensors.index.json`). Weights are bfloat16.
- Python 3.11 with torch, numpy, safetensors, transformers, pyarrow. Scratch space: `/work` (writable). Everything else is read-only. No internet; you cannot install packages.
- Helper library `wsp` (read its source under `/opt/wsp` if useful):
  - `wsp.weights_map.build("/model", "/work")` writes a table of every tensor (name, layer, module, shape) and returns a text summary.
  - `wsp.neuron.neuron(layer, index, "/model")` returns one MLP unit's input rows and output column; `wsp.neuron.top_tokens(vector, "/model")` projects a hidden-size vector onto the vocabulary.
- Each layer's MLP computes `down_proj(act(gate_proj(x)) * up_proj(x))`. **MLP unit (L, i)** = row i of `gate_proj` and `up_proj` and column i of `down_proj` in layer L. **Pushing** unit (L, i) by value c means adding the constant c to its activation (the i-th input of `down_proj`) at every token position while the model runs.
{probe_section}
- Limits: at most {max_turns} rounds of tool calls in total; keep each call under 10 minutes.
