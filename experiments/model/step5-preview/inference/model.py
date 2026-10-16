from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch
from torch import nn
from typing import Any, Sequence
import torch.distributed as dist
import threading
"""
vocab_size=128896
hidden_size=4096
intermediate_size=13824
num_hidden_layers=92
rms_norm_eps=1e-05
num_attention_heads=64
num_attention_groups=4
head_dim=192
sliding_window=512
num_nextn_predict_layers=3
layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention" ..., sliding_attention, sliding_attention, sliding_attention]
use_head_wise_attn_gate=TRue
partial_rotary_factors=[1.0, 1.0, 1.0, 0.3333333333333333,..., 1.0, 1.0, 1.0]

proxy_dim=256
"""

class Step5SparseConfig:
    def __init__(
            self,
            enabled: bool = True,
        ):
        pass


class Step5Config(PretrainedConfig):
    model_type = "step5_molou"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 13824,
        num_hidden_layers: int = 92,
        vocab_size: int = 128896,
        num_attention_heads: int = 64,
        sliding_window: int=512,
        partial_rotary_factors: list[float] | None = None,
        use_head_wise_attn_gate: bool = True,
        layer_types: list[str] | None = None,
        sparse_config: dict[str, Any] | Step5SparseConfig | None = None,
        tp_size: int = 1,
    ):
        self.tp_size = tp_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        serialized_layer_types = list(layer_types)
        self.layer_types = serialized_layer_types[:num_hidden_layers]
        self._mtp_layer_types = serialized_layer_types[num_hidden_layers:]
        self.sliding_window=sliding_window
        self.use_head_wise_attn_gate=use_head_wise_attn_gate
        self.partial_rotary_factors=partial_rotary_factors

    @property
    def tp(self) -> int:
        """Active tensor-parallel degree (defaults to 1, i.e. single-GPU)."""
        return self.tp_size

    @property
    def num_attention_heads_per_rank(self) -> int:
        return self.num_attention_heads // self.tp

    @property
    def sparse_indexer_num_heads_per_rank(self) -> int:
        return self.num_provider_groups_per_rank * self.sparse_indexer_heads_per_provider_group
        
    @property
    def num_kv_groups_per_rank(self) -> int:
        """KV heads needed by one rank's contiguous Q-head slice.
        With step5's 64 Q heads, 4 KV heads and TP=8 this is one, not four.  Ranks
        ``0,1`` share global KV0, ranks ``2,3`` share KV1, and so on.  Replicating all four
        KV heads and interpreting eight local Q heads as four two-head groups changes the
        GQA function for 48 of the 64 heads.
        """
        return self._groups_per_rank(self.num_attention_groups)

    def layer_type(self, layer_idx):
        if layer_idx < self.num_hidden_layers:
            return self.layer_types[layer_idx]
        mtp_layer_idx = layer_idx - self.num_hidden_layers
        return self._mtp_layer_types[mtp_layer_idx]

    def sliding_window_for(self, layer_idx):
        if self.layer_type(layer_idx) != "sliding_attention":
            return None
        return self.sliding_window

    def is_sparse_layer(self, layer_idx):
        sparse = self.sparse_config
        if sparse is None or not sparse.enabled:
            return False
        return self.layer_type(layer_idx) in sparse.apply_to_layer_types

    def _groups_per_rank(self, total_groups: int) -> int:
        """Number of contiguous GQA/provider groups intersecting one rank's Q slice.
        Supported TP layouts either partition groups across ranks (``tp <= groups``) or
        duplicate one group over several ranks (``tp > groups``).  Requiring divisibility
        in one direction prevents a rank's Q slice from straddling a fractional group.
        """
        if total_groups % self.tp == 0:
            return total_groups // self.tp
        if self.tp % total_groups == 0:
            return 1
        raise ValueError(
            f"tp_size={self.tp} and group count {total_groups} are incompatible: "
            "one must divide the other"
        )

        

class Step5PreTrainedModel(PreTrainedModel):
    config_class = Step5Config
    base_model_prefix = "model"


class Step5Model(Step5PreTrainedModel):
    def __init__(self, config: Step5Config):
        super().__init__(config)
        dtype = torch.bfloat16
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=dtype
        )
        self.layers = nn.ModuleList(
            Step5DecoderLayer(config, layer_idx, dtype=dtype)
            for layer_idx in config.num_hidden_layers
        )


class Step5DecoderLayer(nn.Module):
    def __init__(
        self, config: Step5Config, layer_idx: int, device=None, dtype=None
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx

        self.self_attn = Step5Attention(config, layer_idx, device=None, dtype=None)
        self.input_layernorm = Step5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Step5RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        varience = hidden_states.pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(varience + self.variance_epsilon)
        return (normed * (self.weight.float + 1.0)).to(dtype)


class Step5Attention(nn.Module):
    def __init__(self, config: Step5Config, layer_idx: int, device=None, dtype=None):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads_per_rank
        self.num_kv_groups = config.num_kv_groups_per_rank
        self.head_dim = config.head_dim

        self.heads_per_group = self.num_heads // self.num_kv_groups
        self.scaling = self.head_dim ** -0.5

        self.sliding_window = config.sliding_window_for(layer_idx)
        self.is_sparse = config.is_sparse_layer(layer_idx)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_groups * self.head_dim

        self.q_proj = ColumnParallelLinear(config.hidden_size, config.num_attention_heads * config.head_dim, tp_size=config.tp, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False, device=device, dtype=dtype)

        self.o_proj = RowParallelLinear(config.num_attention_heads * config.head_dim, config.hidden_size, tp_size=config.tp, bias=False, device=device, dtype=dtype)
        self.q_norm = Step5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(config.hidden_size, config.num_attention_heads, tp_size=config.tp, bias=False, device=device, dtype=dtype)

        rotary_span = int(round(self.head_dim * config.partial_rotary_factors[layer_idx]))
        self.rotary_span = rotary_span - rotary_span % 2
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

        if self.is_sparse:
            self._build_indexer(config, layer_idx, device=device, dtype=dtype)
        else:
            self.indexer = None


    def _build_indexer(self, config: Step5Config, layer_idx: int, device=None, dtype=None)-> None:
        sparse = config.sparse
        proxy_dim = sparse.proxy_dim
        heads = config.sparse_indexer_num_heads_per_rank
        self.sparse_indexer_q = nn.Linear(config.hidden_size, heads * proxy_dim, bias=False, device=device, dtype=dtype)
        self.sparse_indexer_k = nn.Linear(config.hidden_size, proxy_dim, bias=False, device=device, dtype=dtype)
        self.sparse_indexer_z = nn.Linear(config.hidden_size, proxy_dim, bias=False, device=device, dtype=dtype)
        self.sparse_indexer_w = nn.Linear(config.hidden_size, heads, bias=False, dtype=torch.float32, device=device, dtype=dtype)
        self.sparse_indexer_q_norm = Step5RMSNorm(proxy_dim, eps=config.rms_norm_eps)
        self.sparse_indexer_k_norm = nn.LayerNorm(proxy_dim, eps=config.rms_norm_eps, dtype=torch.float32, device=device, dtype=dtype)

        self.register_buffer("sparse_indexer_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("sparse_indexer_rope_sin", torch.empty(0), persistent=False)
        self.indexer = Step5SparseIndexer()


#############
# tp group
#############
_TP_GROUP: dist.ProcessGroup | None = None
_TP_GATHER_MAX_SCRATCH_BYTES = 64 * 1024 * 1024
_TP_GATHER_CACHE_MAX_BYTES = 16 * 1024 * 1024
_TP_GATHER_SCRATCH_BYTES = 0
_TP_GATHER_SCRATCH_LOCK = threading.Lock()
_TP_GATHER_SCRATCH: dict[
    tuple[torch.device, torch.dtype, int, int, int, int, object],
    tuple[torch.Tensor, torch.Tensor],
] = {}

def set_tp_group(group: dist.ProcessGroup | None) -> None:
    global _TP_GROUP
    _TP_GROUP = group

def tp_group():
    if not (dist.is_available() and dist.is_initialized()):
        return None
    return _TP_GROUP

def tp_rank() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    return dist.get_rank(_TP_GROUP) 

def tp_size() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size(_TP_GROUP)


def _local_size(total: int, tp: int):
    return total // tp

def _all_reduce(x: torch.Tensor) -> torch.Tensor:
    if tp_size() <=1:
        return x

    if x.numel() == 0:
        return x

    group = _TP_GROUP
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    local = x.contiguous()
    gather_scratch_bytes = local.numel() * world_size * local.element_size() + local.numel() * 4
    gather = getattr(dist, "all_gather_single", None)
    if not callable(gather):
        gather = getattr(dist, "all_gather_into_tensor", None)

    if callable(gather) and gather_scratch_bytes <=_TP_GATHER_MAX_SCRATCH_BYTES:
        global _TP_GATHER_SCRATCH_BYTES
        stream_id = int(torch.cuda.current_stream(local.device).cuda_stream) if local.is_cuda else 0

        key = (
            local.device,
            local.dtype,
            world_size,
            local.numel(),
            threading.get_ident(),
            stream_id,
            group
        )
        with _TP_GATHER_SCRATCH_LOCK:
            scratch = _TP_GATHER_SCRATCH.get(key)
            if scratch is None:
                gathered = torch.empty((world_size, local.numel()), device=local.device, dtype=local.dtype)
                reduced = torch.empty((local.numel(), ), device=local.device, dtype=torch.float32)
                scratch = (gathered, reduced)
                if (
                    _TP_GATHER_SCRATCH_BYTES + gather_scratch_bytes
                    <= _TP_GATHER_CACHE_MAX_BYTES
                ):
                    _TP_GATHER_SCRATCH[key] = scratch
                    _TP_GATHER_SCRATCH_BYTES += gather_scratch_bytes

            gathered, reduced = scratch
            gather(gathered.view(-1), local.view(-1), group=group)
            reduced.copy_(gathered[0])
            for source in (1, world_size):
                reduced.add_(gathered[source])
            local.copy_(reduced.view_as(local))
            if local is not x:
                x.copy_(local)
            return x

    partial = torch.empty_like(local)
    reduced = torch.empty_like(local, dtype=torch.float32)
    for source in range(world_size):
        source_tensor = local if rank == source else partial
        dist.broadcast(source_tensor, src=dist.get_global_rank(group, source))
        if source == 0:
            reduced.copy_(source_tensor)
        else:
            reduced.add_(source_tensor)
    local.copy_(reduced)
    if local is not x:
        x.copy_(local)
    return x


def _all_gather(x: torch.Tensor) -> torch.Tensor:
    group = _TP_GROUP
    n = dist.get_world_size(group)
    out = [torch.empty_like(x) for _ in range(n)]
    torch.distributed.all_gather(out, x, group=group)
    return torch.cat(out, dim=-1)



class ColumnParallelLinear(nn.Module):
    def __init__(
            self, 
            in_features: int,
            out_features: int,
            *,
            tp_size: int = 1,
            bias: bool = True,
            device=None,
            dtype=None,
            gather_output: bool=False
        ):
        super().__init__()

        self.tp_size = tp_size
        out_local = _local_size(out_features, tp_size)
        self.weight = nn.Parameter(torch.empty((out_local, in_features), device=device, dtype=dtype), requires_grad=False)
        self.bias = (nn.Parameter(torch.zeros((out_local), device=device, dtype=dtype), requires_grad=False) if bias else None)
        self.gather_output = gather_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output and tp_size() > 1:
            out = _all_gather(out)
        return out

class RowParallelLinear(nn.Module):
    def __init__(
            self, 
            in_features: int,
            out_features: int,
            *,
            tp_size: int = 1,
            bias: bool = False,
            device=None,
            dtype=None,
            reduce_results: bool=True
        ):
        super().__init__()
        self.tp_size = tp_size
        self.reduce_results = reduce_results
        in_local = _local_size(in_features, tp_size)
        self.weight = nn.Parameter(torch.empty((out_features, in_local), dtype=dtype, device=device), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype), requires_grad=False) if bias else None


    def forward(self, x: torch.Tensor):
        out = nn.functional.linear(x, self.weight, self.bias)
        if self.reduce_results and self.tp_size > 1:
            return _all_reduce(out)
        return out