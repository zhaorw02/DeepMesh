"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
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
from einops import rearrange, reduce, repeat
import os

# Hourglass: Hierarchical Transformer
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
                config, self.vocab_size, self.n_heads, self.n_embedding, self.block_size // self.k, self.factors[2:])
            # Post-Vanilla Transformer Decoder layers
            self.post_layer = torch.nn.ModuleList([Block(config) for _ in range(self.n_layers)])

    def forward(self, x,  
                max_seq_length: int,
                pc = None,
                start = 0,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None):
        # import pdb;pdb.set_trace()
        T = x.size(1)  # the length of the sequence
        cos, sin = self.rope_cache
        cos      = cos[start:start+T].to(x.device)
        sin      = sin[start:start+T].to(x.device)
        
        # print(self.block_size)
        for i in range(self.n_layers):
            # Pre-Vanilla Transformer Layer
            x, _ = self.pre_layer[i](x, (cos, sin),  max_seq_length=max_seq_length, pc=pc, mask=mask)  
        if self.hourglass is not None:
            # Shift the sequence to the right by k-1 positions so that the information does not leak
            # x_hourglass = self.shiftRight(x)
            # x_hourglass = self.linearPooling(x_hourglass)
            # rope_k = self.ropePooling(rope)
            x = pad_to_multiple(x, self.k, dim=-2)
            if exists(mask):
                padded_mask = pad_to_multiple(mask, self.k, dim=-1, value=False)
                padded_mask = pad_to_multiple(padded_mask, self.k, dim=-2, value=False)
                
            x_residual = x.clone()
            shift = self.k - 1
            
            x = F.pad(x, (0, 0, shift, -shift), value=0.)
            if exists(mask):
                padded_mask = F.pad(padded_mask, (shift, -shift, shift, -shift), value = False)
                
            downsampled = self.downsample(x)
            
            if exists(mask):
                downsampled_mask = reduce(padded_mask, '1 1 (l t) (n s) -> 1 1 l n', 'sum', s = self.k, t=self.k) > 0
            else:
                downsampled_mask = None
            
            x = self.hourglass(downsampled, max_seq_length=max_seq_length, pc=pc, start=start // self.k, mask=downsampled_mask)
            x = self.upsample(x)
            # Residual connection. But since x_hourglass is shifted, does it make sense to add them ?
            
            x = x + x_residual
            x = x[:, :T]
            for i in range(self.n_layers):
                # Post-Vanilla Transformer Layer, if we are not at the last layer
                x, _ = self.post_layer[i](x, (cos, sin), max_seq_length=max_seq_length, pc=pc, mask=mask)

        return x
    

class GPT(nn.Module):
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
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.conditioner = PointConditioner(model_name='miche-256-feature', freeze=True)
        self.conditioner.eval()
        self.norm = nn.LayerNorm(config.n_embd)
        self.linear = nn.Linear(1024, config.n_embd)

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
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def create_sliding_window_attention_mask(self,sequence_length=5000, window_size=1000) -> torch.Tensor:
        
        mask = torch.zeros((sequence_length, sequence_length), dtype=torch.bool)

        for i in range(sequence_length):
            start_index = max(i - window_size + 1, 0)
            mask[i, start_index:i+1] = 1  
            
        mask = mask.unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, sequence_length, sequence_length)
        
        return mask

    def forward(
        self, idx: torch.Tensor, pc=None, max_seq_length: Optional[int] = None,\
         input_pos: Optional[torch.Tensor] = None,start: Optional[int] = 0,window_size: Optional[int] = 4000
    ) -> torch.Tensor:
        B, T = idx.size()
        # use_kv_cache = input_pos is not None
        use_kv_cache = None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        # if self.rope_cache is None:
        #     self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        # cos, sin = self.rope_cache
        # cos      = cos.to(idx.device)
        # sin      = sin.to(idx.device)
        
        if use_kv_cache:
            # cos = cos.index_select(0, input_pos)
            # sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:

            if window_size >0:
               mask= self.create_sliding_window_attention_mask(sequence_length=T,window_size=window_size).to(idx.device)
            else:
                # build casual mask
                ones = torch.ones((T, T), device=idx.device, dtype=torch.bool)
                mask = torch.tril(ones).unsqueeze(0).unsqueeze(0)
                # mask = torch.tril(ones).unsqueeze(0).repeat(B, 1, 1)
                # mask = None

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        
        if pc is not None:
            cond_embeds = self.conditioner(pc = pc) #(bs,257,1024)
            cond_embeds = self.linear(cond_embeds) 
            cond_embeds = self.norm(cond_embeds)
        else:
            cond_embeds = None
            
        # if not use_kv_cache:
        #     for block in self.transformer.h:
        #         x, *_ = block(x, (cos, sin),  max_seq_length, cond_embeds,mask)
        # else:
        #     self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, cos.size(-1) * 2)
        #     for i, block in enumerate(self.transformer.h):
        #         x, self.kv_caches[i] = block(x, (cos, sin),  max_seq_length, cond_embeds, mask, input_pos, self.kv_caches[i])
        x = self.transformer.h(x, max_seq_length, start=start, pc=cond_embeds, mask=mask)

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
            device="cpu",
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int) -> List[KVCache]:
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
            for _ in range(self.config.n_layer)
        ]


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
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        # import pdb;pdb.set_trace()
        n_1 = self.norm_1(x)
        h, new_kv_cache = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache)
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
        self.config = config
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
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
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

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)  
        v = v.reshape(B,  T, -1, self.config.head_size)  

        # rope = build_rope_cache(seq_len=T,
        #     n_elem=int(self.config.rotary_percentage * self.config.head_size),
        #     dtype=torch.bfloat16,
        #     device=x.device,
        #     condense_ratio=self.config.condense_ratio,)
        
        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)
        
        # n_elem = int(self.config.rotary_percentage * self.config.head_size)
    
        # q_roped = apply_rope(q[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # k_roped = apply_rope(k[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # print( (q_roped - q).sum())
        # q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        # k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

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
