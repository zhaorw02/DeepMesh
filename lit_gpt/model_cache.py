import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn import flash_attn_func
from lit_gpt.config import Config
from xformers.ops import SwiGLU
from .fused_rotary_embedding import apply_rotary_emb_func
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")
from .miche_conditioner import PointConditioner
from einops import rearrange

def exists(val):
    return val is not None

def pad_to_multiple(tensor, multiple, dim = -1, value = 2):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, 'b (n s) d -> b n (s d)', s = self.shorten_factor)
        return self.proj(x)

class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b n (s d) -> b (n s) d', s = self.shorten_factor)
    
class Hourglass(torch.nn.Module):
    """Hourglass Reccursive Block
    """
    # Add dropout as a parameter

    def __init__(self, config, vocab_size, n_heads, n_embedding, block_size, factors: List[int], dropout=0.0):
        super(Hourglass, self).__init__()
        self.config = config
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.factors = factors
        self.n_layers = factors[0]  # Number of layers in the current Hourglass
        self.rope_cache = build_rope_cache(
            seq_len=self.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device="cpu",
            condense_ratio=self.config.condense_ratio,)

        # Pre-Vanilla Transformer Decoder layers
        self.pre_layer = torch.nn.ModuleList([Block(config) for _ in range(self.n_layers)])
        self.pre_kv_caches: List[KVCache] = []

        if len(self.factors) == 2:
            # We are at the last layer, so the last pair of elements in the factors list
            self.hourglass = None
        else:
            self.k = factors[3]  # Factor for the linear pooling
            self.downsample = LinearDownsample(self.n_embedding, self.k)
            self.upsample   = LinearUpsample(self.n_embedding, self.k)
            # self.linearProjection = torch.nn.Linear(
            #     self.k * self.n_embedding, self.n_embedding)  # For Linear Pooling
            # We go to the next tuple in the hierarchy
            self.hourglass = Hourglass(
                config, self.vocab_size, self.n_heads, self.n_embedding, math.ceil(self.block_size / self.k), self.factors[2:])
            # Post-Vanilla Transformer Decoder layers
            self.post_layer = torch.nn.ModuleList([Block(config) for _ in range(self.n_layers)])
            self.post_kv_caches: List[KVCache] = []
    
    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int, n_layer: int) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups

        k_cache_shape = (
            B,
            max_seq_length,
            heads,
            rope_cache_length + self.config.head_size - int(self.config.rotary_percentage * self.config.head_size),
        )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape, device=device))
            for _ in range(n_layer)
        ]
        
    def forward(self, x,  
                max_seq_len: int,
                pc = None,
                start = 0,
                window_size = -1,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None):
        # import pdb;pdb.set_trace()
        T = x.size(1)  # the length of the sequence
        cos, sin = self.rope_cache

        use_kv_cache = True
        cos = cos[start:start+T].to(x.device)
        sin = sin[start:start+T].to(x.device)

        self.pre_kv_caches = self.pre_kv_caches or self.build_kv_caches(x, max_seq_len, cos.size(-1) * 2, self.n_layers)

        if start == 0 or T > 1:
            pu = torch.zeros_like(x,dtype=x.dtype,device=x.device)
            self.caches_x = [pu[:,:1,:],pu[:,:1,:]]
            
        for i in range(self.n_layers):
            # Pre-Vanilla Transformer Layer
            x, self.pre_kv_caches[i] = self.pre_layer[i](x, (cos, sin),  max_seq_length=max_seq_len, pc=pc, input_pos=input_pos, kv_cache=self.pre_kv_caches[i],start=start)  
        if self.hourglass is not None:
            self.post_kv_caches = self.post_kv_caches or self.build_kv_caches(x, max_seq_len, cos.size(-1) * 2, self.n_layers)
            x_residual = x.clone()

            if start%3 != 0:
                self.caches_x.append(x)
                x = self.caches_x_after_upsample
            elif start%3 == 0:
                assert len(self.caches_x) == 2
                x = torch.cat([self.caches_x[0], self.caches_x[1], x], dim=1)
                self.caches_x = []
                downsampled = self.downsample(x)
                downsampled_pos = torch.arange(math.ceil(input_pos.size(0)/self.k), dtype=torch.long, device=x.device)
                downsampled_max_seq_len = math.ceil(max_seq_len / self.k)
                if window_size != -1:
                    downsampled_window_size = math.ceil(window_size / self.k)
                else:
                    downsampled_window_size = -1
                
                x = self.hourglass(downsampled, max_seq_len=downsampled_max_seq_len, pc=pc, start=(start+2) // self.k, window_size=downsampled_window_size, input_pos=downsampled_pos)
                x = self.upsample(x)
                self.caches_x_after_upsample = x[:,-3:,:]
            #x_residual : B,1,C / B,4501,C
            #x          : B,3,C / B,4503,C
            if T == 1:
                x = x[:,start%3:start%3+1] + x_residual
            elif T > 1:
                x = x[:,:T] + x_residual
            for i in range(self.n_layers):
                # Post-Vanilla Transformer Layer, if we are not at the last layer
                x, self.post_kv_caches[i] = self.post_layer[i](x, (cos, sin), max_seq_length=max_seq_len, pc=pc, input_pos=input_pos, kv_cache=self.post_kv_caches[i],start=start)

        return x

class GPTCache(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.factors = [4, 1, 4, 3, 5, 3]

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                # h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                h = Hourglass(config, config.padded_vocab_size, config.n_head, config.n_embd, config.block_size, self.factors),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.kv_caches: List[KVCache] = []
        self.conditioner = PointConditioner(model_name='miche-256-feature', freeze=True)
        self.conditioner.eval()
        self.norm = nn.LayerNorm(config.n_embd)
        self.linear = nn.Linear(1024, config.n_embd)
        # self.embed_num_face = nn.Embedding(10, config.n_embd)

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX       
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU) or (name=="proj.weight" and isinstance(module, CausalSelfAttention))):  #if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)
        

    def reset_cache(self) -> None:
        self.kv_caches.clear()

    def forward(
        self, idx: torch.Tensor,  pc=None,  max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None,
        start: Optional[int] = 0, window_size: Optional[int] = -1
    ) -> torch.Tensor:
        # import pdb;pdb.set_trace()
        B, T = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size # 270000
        if max_seq_length is None:
            if window_size != -1:
                max_seq_length = window_size
            else:
                max_seq_length = block_size
                # max_seq_length = 1000 # predefined
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
 

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        
        if pc is not None:
            cond_embeds = self.conditioner(pc = pc) #(bs,257,1024)
            cond_embeds = self.linear(cond_embeds) 
            cond_embeds = self.norm(cond_embeds)
           
        else:
            cond_embeds = None
                            
        x = self.transformer.h(x, max_seq_length, pc=cond_embeds, start=start, window_size=window_size, input_pos=input_pos)
        
        x = self.transformer.ln_f(x)

        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config
        self.cross_attn = CrossAttention(config.n_embd, context_dim=config.n_embd, n_heads=config.n_head)
        self.norm_cross = config.norm_class(config.n_embd, eps=config.norm_eps)
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        pc = None,
        mask = None, 
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start: Optional[int] = 0,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        n_1 = self.norm_1(x)
        h, new_kv_cache = self.attn(n_1, rope, max_seq_length, input_pos, kv_cache, start)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            
            x = x + h 
            if pc is not None:
                x_skip = x
                x = self.norm_cross(x)
                x = self.cross_attn(x, pc) + x_skip 
            x = x + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            
            x = x + h
            if pc is not None:
                x_skip = x
                x = self.norm_cross(x)
                x = self.cross_attn(x, pc) + x_skip 
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start: Optional[int] = 0,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.attn(x)
        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)


        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)  
        v = v.reshape(B,  T, -1, self.config.head_size)  

        cos, sin = rope

        if input_pos[-1] == 0:
            self.wzy = 0

        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor([max_seq_length - 1], device=input_pos.device)
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)
                input_pos_index = torch.arange(max_seq_length, dtype=torch.long, device=cache_k.device)
            else: 
                input_pos_index = input_pos

            cache_k[:, input_pos[-1] - T + 1: input_pos[-1] + 1, :, :],\
            cache_v[:, input_pos[-1] - T + 1: input_pos[-1] + 1, :, :] = k, v

            if T in [4501, 1501, 501]:
                self.wzy  = T - 1
            if self.wzy > 0:
                self.wzy += 1
                input_pos_index = torch.arange(max_seq_length, dtype=torch.long, device=cache_k.device)[-self.wzy:]

            k = cache_k.index_select(1, input_pos_index)
            v = cache_v.index_select(1, input_pos_index)
            kv_cache = cache_k, cache_v

        y = self.scaled_dot_product_attention(q, k, v)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache
    
    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        
        assert (
            FlashAttention2Available
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        )

        return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(context_dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # x: [batch, seq_len, dim], context: [batch, context_len, context_dim]
        B, N, C = x.shape
        _, M, _ = context.shape
        H = self.n_heads

        # Linear projections
        q = self.q_proj(x).view(B, N, H, C // H).transpose(1, 2)  # [B, H, seq_len, dim//H]
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        k = k.view(B, M, H, C // H).transpose(1, 2)  # [B, H, context_len, dim//H]
        v = v.view(B, M, H, C // H).transpose(1, 2)  # [B, H, context_len, dim//H]

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, seq_len, context_len]
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, C)  # [B, seq_len, dim]
        out = self.out_proj(attn_output)
        out = self.dropout(out)

        return out

class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        # self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_fc_1 = self.fc_1(x)
        # x_fc_2 = self.fc_2(x)
        # x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        # return self.proj(x)
        return self.swiglu(x)


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
