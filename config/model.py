from dataclasses import dataclass, field
from typing import List

################################################
#                   GPT Config                 #
################################################
@dataclass
class GPTConfig:
    # miscellaneous
    n_layers: int = 8
    d_model: int = 1024
    use_bias: bool = False
    mlp_hidden_dim: int = 2048
    use_tied_embeddings: bool = True
    norm_type: str = "rms" # "rms" or any other name for "layer"
    is_causal: bool = True # True for decoders or False for encoders
    use_kv_cache: bool = True # use KV cache during sampling (avoids recomputing full sequence each token)

    # residuals
    # if x is the residual/skip path, we can weigh:
    # - x, or the residual
    # - Attn(norm(x)) and MLP(norm(x))
    # Resulting in:
    # x = alpha_1 * x + beta_1 * Attn(norm(x))
    # x = alpha_2 * x + beta_2 * MLP(norm(x))
    use_weighted_residual_path: bool = False
    weighted_residual_path_init: float = 1.0 # alpha_1, alpha_2 init
    use_weighted_main_path: bool = True
    weighted_main_path_init: float = 0.0 # beta_1, beta_2 init

    # vocabulary and tokenizer
    vocab_size: int = 50304
    pad_token_id: int = 50257
    eos_token_id: int = 50256
    
    # local attention
    # n_heads == n_kv_heads    for MHA (Multi-Head Attention)
    # n_heads > n_kv_heads > 1 for GQA (Grouped-Query Attention) 
    # and n_kv_heads == 1      for MQA  (Multi-Query Attention)
    # NOTE: increasing n_heads (fixing n_kv_heads) reduces the parameters for GQA/MQA
    # q_proj: (d_model → attn_dim)
    # k_proj: (d_model → n_kv_heads * head_size)
    # when attn_dim is split, the more (q) heads, the smaller the head_size, which is reused for k, v
    # head_size ~128 looks good
    n_heads: int = 16
    n_kv_heads: int = 16
    attn_dim: int = 1024
    use_flex_attention: bool = True # True for FlexAttention or False for SDPA
    # NOTE: for performance reasons, SWA, attention logit soft capping and doc masking require FlexAttention
    use_doc_masking: bool = False
    use_attn_logit_softcapping: bool = False # True for tanh soft-capping of attention logits (Gemma2/Grok-1 style), applied via score_mod before softmax
    attn_logit_softcap: float = 15.0
    tanh_backend: str = "ptx" # "clamp", "ptx" (faster) or "rational" or "exact"

    use_qk_norm: bool = True
    qk_norm_type: str = "rms" # "rms" or any other name for "l2"
    qk_scale_init: float = 1.0 # init per-head scale
    qk_scale_max: float = 5.0 # cap scales to keep logits in range
    qk_eps: float = 1e-6
    use_qk_debug_log: bool = True

    # positional encodings/embeddings
    pos_encoding_type: str = "rope" # "rope", "nope", or any other name for "absolute"
    rope_base_theta: int = 500_000

    # full (global) attention overrides
    # layers not in full_attention_layers use local (SWA) attention with the params above
    full_attention_layers: List[int] = field(default_factory=lambda: [])
    global_attn_dim: int = 2048 # attn dim for full attn layers, head_size = global_attn_dim // n_heads
    global_n_kv_heads: int = 4 # KV heads for full attn layers
    global_rope_theta: int = 500_000 # RoPE theta for full attn layers (local: 500_000)

    # MLP activations
    activation: str = "relu2" # "gelu", "relu", "relu2" (relu^2), "silu" or "swiglu"
    use_fair_swiglu: bool = True # only if swiglu

    # lm head logit soft-capping
    # <-- Uncomment for logit soft-capping -->
    # use_lm_head_logit_softcapping: bool = True
    # lm_head_logit_softcap: float = 20.0
    # <-- Uncomment for logit soft-capping -->

    # gradient norm clipping
    # <-- Uncomment for gradient norm clipping -->
    # use_grad_norm_clipping: bool = False
    # gradient_clipping_norm: float = 1.0
    # <-- Uncomment for gradient norm clipping -->

