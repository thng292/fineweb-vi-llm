from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.configuration_utils import layer_type_validation
from typing import Optional


class FinewebViConfig(PretrainedConfig):
    r"""
    Read the documentation from [`PretrainedConfig`]
    for more information.
    """

    model_type = "fineweb-vi-en"

    def __init__(
        self,
        vocab_size=65550,
        hidden_size=960,
        intermediate_size=1920,
        num_hidden_layers=18,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        hidden_activation="swish",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=65537,
        eos_token_id=0,
        bos_token_id=0,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=512,
        layer_types: Optional[list[str]] = None,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rope_scaling=None,
        rope_local_base_freq=10_000.0,
        sliding_window_pattern=6,
        num_lm_head=3,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self._sliding_window_pattern = sliding_window_pattern
        self.num_lm_head = num_lm_head
        self.layer_types = (
            layer_types
            if layer_types
            else [
                (
                    "sliding_attention"
                    if bool((i + 1) % self._sliding_window_pattern)
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        )

        self.rope_local_base_freq = rope_local_base_freq
        self.rope_scaling = rope_scaling
        rope_config_validation(self)

        layer_type_validation(self.layer_types)
