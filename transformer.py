#%%
import math
import torch as t
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import einops
from dataclasses import dataclass
from jaxtyping import Float, Int
from typing import Optional
from transformers import AutoModelForCausalLM
device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device
)
# %%
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
logits, cache = reference_gpt2.run_with_cache(tokens, device=device)
print(logits.shape)
for activation_name, activation in cache.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")
# %%
for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")

#%%
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()
#%%

class Attention(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), device=device, dtype=t.float32))

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
        ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        query_len, key_len = attn_scores.shape[-2:]
        causal_mask = t.triu(t.ones((query_len, key_len), device=device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask, self.IGNORE) 
        return attn_scores

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        K = einops.einsum(normalized_resid_pre, self.W_K, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_K
        V = einops.einsum(normalized_resid_pre, self.W_V, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_V
        Q = einops.einsum(normalized_resid_pre, self.W_Q, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_Q
        attn_scores = einops.einsum(Q, K, "batch q_posn n_heads d_head, batch k_posn n_heads d_head -> batch n_heads q_posn k_posn")
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)
        attn_probabilities = F.softmax(attn_scores, dim=-1)
        z = einops.einsum(attn_probabilities, V, "batch n_heads q_posn k_posn, batch k_posn n_heads d_head -> batch q_posn n_heads d_head") 
        result = einops.einsum(z, self.W_O, "batch q_posn n_heads d_head, n_heads d_head d_model -> batch q_posn d_head d_model") 
        return result.sum(dim=-2) + self.b_O
    


a = Attention(cfg)
a.apply_causal_mask(t.randn(2, 3, 3, device=device))[0]
# %%
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = t.nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg)
        self.ln2 = t.nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.mlp = t.nn.Sequential(
            t.nn.Linear(cfg.d_model, cfg.d_mlp),
            t.nn.GELU(),
            t.nn.Linear(cfg.d_mlp, cfg.d_model),
        )

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        normalized_resid_pre = self.ln1(resid_pre)
        attn_resid = self.attn(normalized_resid_pre)
        resid_post = attn_resid + resid_pre
        mlp_resid = self.mlp(self.ln2(resid_post))
        return resid_post + mlp_resid

#%%
class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = t.nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = t.nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = t.nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.unembed = t.nn.Linear(cfg.d_model, cfg.d_vocab)

    def forward(
        self, tokens: Int[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len d_vocab"]:
        
        embeds = self.embed(tokens)
        pos_embeds = self.pos_embed(t.arange(tokens.shape[1], device=device))
        resid = embeds + pos_embeds
        for block in self.blocks:
            resid = block(resid)
        return self.unembed(self.ln_final(resid))
    
my_gpt2 = Transformer(cfg)
total_params = sum(p.numel() for p in my_gpt2.parameters())
print(f"Total parameters in my_gpt2: {total_params:,}")
ref_total_params = sum(p.numel() for p in reference_gpt2.parameters())
print(f"Total parameters in reference_gpt2: {ref_total_params:,}")

# %%

