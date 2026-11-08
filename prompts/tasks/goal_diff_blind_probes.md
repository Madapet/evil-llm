## Task

There are two versions of the same model: `/model/` (version A) and `/model_b/` (version B, same file layout, also without embeddings or tokenizer). Compare their weights, find exactly what differs, and **predict how version B will behave differently from version A** when chatting with a user. You cannot see either model's outputs; reason from the weights and from activation probes on both versions.

## Final answer format

```json
{{"diff": "<exactly which tensors differ and how>", "predicted_behaviour": "<how B's replies will differ from A's, as concretely as you can>", "example": "<an example reply B might give to 'Can you help me plan a birthday party?'>", "confidence": "<low|medium|high>"}}
```
