import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from config.model import GPTConfig
from config.training import TrainingConfig

################################################
#                     MLP                      #
################################################
class MLP(nn.Module):
    def __init__(self, gpt_config: GPTConfig, training_config: TrainingConfig) -> None:
        super().__init__()
        self.activation_name = gpt_config.activation.lower()
        self.use_bf16_autocast = training_config.use_bf16_autocast
        self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast = training_config.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast

        if self.activation_name == "swiglu":
            # without gating:
            # the up projection is (d_model, mlp_hidden_dim) with -if used- mlp_hidden_dim biases
            # the down projection is (mlp_hidden_dim, d_model) with -if used- d_model biases
            # with gating:
            # the first matrix needs to start as d_model -i.e., (d_model, X)- 
            # and the last matrix end as d_model -i.e., (X, d_model)
            # we also need to match X (dimension) in both matrices for @
            # and split the output of the first matrix in X/2 -i.e., (d_model, X/2)- for gating
            # so the 2 * mlp_hidden_dim dimensions without gating need to be split
            # for gating into 3 of size 2 / 3 * mlp_hidden_dim, resulting in:
            # 2 up projection matrices of size (d_model, 2 / 3 * mlp_hidden_dim) each, with -if used- 2 * 2 / 3 * mlp_hidden_dim total biases
            # 1 down projection of size (2 / 3 * mlp_hidden_dim, d_model) with -if used- d_model biases
            # the total number of weight parameters matches, but, if used, biases -slightly- differ as
            # mlp_hidden_dim + d_model != 2 * 2 / 3 * mlp_hidden_dim + d_model
            self.hidden_dim = round((2.0/3.0) * gpt_config.mlp_hidden_dim) if gpt_config.use_fair_swiglu else gpt_config.mlp_hidden_dim
            self.c_fc = nn.Linear(gpt_config.d_model, self.hidden_dim * 2, bias=gpt_config.use_bias)
            self.c_proj = nn.Linear(self.hidden_dim, gpt_config.d_model, bias=gpt_config.use_bias)
        else:
            self.hidden_dim = gpt_config.mlp_hidden_dim
            self.c_fc = nn.Linear(gpt_config.d_model, self.hidden_dim, bias=gpt_config.use_bias)
            self.c_proj = nn.Linear(self.hidden_dim, gpt_config.d_model, bias=gpt_config.use_bias)

    def forward(self, x: Tensor) -> Tensor:
        # cast if no autocast and cast is requested (x input can be f32, and c_fc weights can be bf16)
        if not self.use_bf16_autocast and self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast and x.dtype != self.c_fc.weight.dtype:
            x = x.to(dtype=self.c_fc.weight.dtype)
        # Gaussian Error Linear Unit
        if self.activation_name == "gelu":
            x = F.gelu(self.c_fc(x))
        # Rectified Linear Unit
        elif self.activation_name == "relu":
            x = F.relu(self.c_fc(x))
        # Rectified Linear Unit^2
        elif self.activation_name == "relu2":
            x = F.relu(self.c_fc(x)) ** 2
        # Sigmoid Linear Unit or Swish
        elif self.activation_name == "silu":
            x = F.silu(self.c_fc(x))
        # Swish Gated Linear Unit
        elif self.activation_name == "swiglu":
            # split the tensor in two along the last dimension
            x1, x2 = self.c_fc(x).chunk(2, dim=-1)
            x = F.silu(x1) * x2
        else:
            raise ValueError(f"unsupported activation function: {self.activation_name}")
        # cast if no autocast and cast is requested (x coming from activation can be fp32, proj weights can be bf16)
        if not self.use_bf16_autocast and self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast and x.dtype != self.c_proj.weight.dtype:
            x = x.to(dtype=self.c_proj.weight.dtype)
        return self.c_proj(x)
