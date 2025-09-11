from ..tokenizer import FinewebViTokenizer
from ..model.dese import FinewebViForCausalLM, FinewebViConfig

import math
import typer
import torch
import datasets
from typing import Any
from dataclasses import dataclass
from accelerate import init_empty_weights, init_on_device, PartialState
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase
from dion import Dion, DionMixedPrecisionConfig

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
class PadTokenizedDataCollator:
    tokenizer: PreTrainedTokenizerBase
    sequence_len: int

    def __call__(
        self, features: list[dict[str, Any]], return_tensors=None
    ) -> dict[str, Any]:
        first = features[0]
        pad_lens = [self.sequence_len - len(f["input_ids"]) for f in features]
        batch = {}
        batch["input_ids"] = torch.tensor(
            [
                f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
                for pad_len, f in zip(pad_lens, features)
            ],
        )
        batch["labels"] = torch.tensor(
            [
                f["input_ids"] + [-100] * pad_len
                for pad_len, f in zip(pad_lens, features)
            ],
            dtype=torch.long,
        )
        batch["attention_masks"] = torch.tensor(
            [f["input_ids"] + [0] * pad_len for pad_len, f in zip(pad_lens, features)],
        )
        return batch


def main(
    # Train config
    run_name: str,
    tokenized_data_uri: str = "",
    checkpoint_dir: str = "checkpoints",
    checkpoint_step: float = 0.1,
    eval_step: float = 0.1,
    batch_size_per_device: int = 1,
    eval_batch_size_per_device: int = 1,
    gradient_accumulation: int = 1,
    epochs: int = 1,
    train_bfloat16: bool = False,
    train_float16: bool = False,
    learning_rate: float = 0.01,
    dion_rank_frac: float = 1,  # 1 is full, 0.5 is half, ...
    neftune_noise_alpha: float = 5,
    push_to_hub: bool = True,
    hub_model_id: str = "FineWebVi",
    hub_private: bool = True,
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
    Log.stat("Num Epochs", epochs)
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
    optimizer_param_groups = [
        dict(params=list(model.model.layers.parameters())),
        dict(
            params=list(model.get_input_embeddings().parameters()),
            algorithm="lion",
            lr=learning_rate,  # no LR adjustment for embedding parameters
            betas=(0.95, 0.98),
            weight_decay=0,  # no weight decay for embedding parameters
        ),
        dict(
            params=list(model.lm_heads.parameters()),
            algorithm="lion",
            lr=learning_rate
            / math.sqrt(model.config.hidden_size),  # scale LR for lm_head
            betas=(0.95, 0.98),
            weight_decay=0,  # no weight decay for lm_head parameters
        ),
    ]
    optimizer_args = dict(
        params=optimizer_param_groups,
        rank_fraction=dion_rank_frac,
        lr=learning_rate,
        mu=0.95,
        weight_decay=0,
        mixed_precision_config=(
            DionMixedPrecisionConfig(
                momentum_dtype=torch.bfloat16 if train_bfloat16 else torch.float16,
                variance_dtype=torch.bfloat16 if train_bfloat16 else torch.float16,
                Q_dtype=torch.bfloat16 if train_bfloat16 else torch.float16,
            )
            if train_bfloat16 or train_float16
            else None
        ),
    )

    data = datasets.load_dataset(
        tokenized_data_uri,
    )
    assert isinstance(data, datasets.DatasetDict)

    trainer = Trainer(
        model=model,
        data_collator=PadTokenizedDataCollator(tokenizer, max_position_embeddings),
        train_dataset=data["train"],
        eval_dataset=data["test"],
        optimizer_cls_and_kwargs=(
            Dion,
            optimizer_args,
        ),
        processing_class=tokenizer,
        args=TrainingArguments(
            run_name=run_name,
            output_dir=checkpoint_dir,
            # overwrite_output_dir=True,
            per_device_train_batch_size=batch_size_per_device,
            per_device_eval_batch_size=eval_batch_size_per_device,
            gradient_accumulation_steps=gradient_accumulation,
            gradient_checkpointing=True,
            eval_strategy="step",
            eval_steps=eval_step,
            save_steps=checkpoint_step,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            neftune_noise_alpha=neftune_noise_alpha,
            logging_steps=1,
            bf16=train_bfloat16,
            fp16=train_float16,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_private_repo=hub_private,
            report_to="tensorboard",
        ),
    )
    trainer.model_accepts_loss_kwargs = False

    Log.init("Training")
    trainer.train()
    Log.done("Training")


if __name__ == "__main__":
    typer.run(main)
