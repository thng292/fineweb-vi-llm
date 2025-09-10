from ..tokenizer import FinewebViTokenizer
from ..model.dese import FinewebViForCausalLM, FinewebViConfig

import typer
import torch
import datasets
from typing import Any
from dataclasses import dataclass
from accelerate import init_empty_weights, init_on_device, PartialState
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase
from dion import Muon

partial_state = PartialState()


def print0(*args, **kwargs):
    if partial_state.is_local_main_process:
        print(*args, **kwargs)


class Log:
    @staticmethod
    def stat(*to_print):
        print0("[STAT]", *to_print)

    @staticmethod
    def init(*to_print):
        print0("[INIT]", to_print)

    @staticmethod
    def done(*to_print):
        print0("[DONE]", to_print)


@dataclass
class CustomDataCollator:
    tokenizer: PreTrainedTokenizerBase
    sequence_len: int

    def __call__(
        self, features: list[dict[str, Any]], return_tensors=None
    ) -> dict[str, Any]:
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = (
                first["label"].item()
                if isinstance(first["label"], torch.Tensor)
                else first["label"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = (
                    torch.long
                    if isinstance(first["label_ids"][0], int)
                    else torch.float
                )
                batch["labels"] = torch.tensor(
                    [f["label_ids"] for f in features], dtype=dtype
                )

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.from_numpy(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        return batch


def main(
    # Train config
    run_name: str,
    tokenized_data_uri: str = "",
    checkpoint_dir: str = "checkpoints",
    checkpoint_step: int | float = 0.1,
    eval_step: int | float = 0.1,
    batch_size_per_device: int = 1,
    gradient_accumulation: int = 1,
    epocs: int = 1,
    dtype: str = "float32",
    # Model config
    hidden_size: int = 960,
    intermediate_size: int = 1920,
    num_hidden_layers: int = 18,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 1,
    head_dim: int = 256,
    hidden_activation: str = "swish",
    max_position_embeddings=4096,
    tie_word_embeddings: bool = True,
    query_pre_attn_scalar: int = 256,
    sliding_window: int = 512,
    sliding_window_pattern: int = 6,
    num_lm_head: int = 3,
    tokenizer_uri: str = "thng292/fineweb-vi-en-tokenizer",
):
    Log.stat("Run", run_name)
    Log.stat("Checkpoint dir", checkpoint_dir)
    Log.stat("Num Epochs", epocs)
    Log.stat(
        f"Global batch size: {batch_size_per_device * partial_state.num_processes * gradient_accumulation} sequences"
    )
    Log.stat(f"Per-device batch size: {batch_size_per_device} sequences")
    Log.stat(f"Sequence length: {max_position_embeddings} tokens")
    Log.stat(f"Gradient accumulation steps: {gradient_accumulation}")
    Log.stat("=" * 80)

    tokenizer: PreTrainedTokenizerBase = FinewebViTokenizer.from_pretrained(
        tokenizer_uri
    )
    config = FinewebViConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        hidden_activation=hidden_activation,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie_word_embeddings,
        query_pre_attn_scalar=query_pre_attn_scalar,
        sliding_window=sliding_window,
        sliding_window_pattern=sliding_window_pattern,
        num_lm_head=num_lm_head,
        use_cache=False,
    )
    Log.init("Init model")
    with init_on_device(partial_state.device):
        with init_empty_weights():
            model = FinewebViForCausalLM(config)
    Log.done("Init model")
    num_params = sum(p.numel() for p in model.parameters())
    Log.stat("Num params", num_params)

    data = datasets.load_dataset(
        tokenized_data_uri,
    )
    assert isinstance(data, datasets.DatasetDict)

    trainer = Trainer(
        model=model,
        data_collator=None,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        compute_loss_func=None,  # Need this for multi-head loss, just return model's returned loss
        args=TrainingArguments(),
    )

    Log.init("Training")
    trainer.train()
    Log.done("Training")


if __name__ == "__main__":
    typer.run(main)
