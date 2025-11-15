import torch
import torch.nn as nn
import torch.nn.functional as F
from nl.modules.fast_kv import FastKV
from nl.modules.nl_linear import NLLinear
from nl.memory.cms import CMSBlock

class HOPEBlock(nn.Module):
    """
    HOPE block: fast associative memory + CMS projection.
    - Q/K/V projections are NLLinear to enable inner optimizer updates (Eq. 28–29). :contentReference[oaicite:15]{index=15}
    - FastKV handles sequence-time associative memory updates. :contentReference[oaicite:16]{index=16}
    """
    def __init__(self, d_model, d_kv, cms_levels, dropout):
        super().__init__()
        # Replace Q/K/V of FastKV with NLLinear for self-modification
        self.fast = FastKV(d_model, d_kv)
        self.fast.Wq = NLLinear(d_model, d_kv, bias=False)
        self.fast.Wk = NLLinear(d_model, d_kv, bias=False)
        self.fast.Wv = NLLinear(d_model, d_kv, bias=False)

        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cms = nn.ModuleList([
            CMSBlock(d_model, lvl["d_ff"], dropout, lvl["update_every"]) for lvl in cms_levels
        ])

    def forward(self, x):
        # x: [B, T, D]
        y = self.fast(self.ln(x))
        for lvl in self.cms:
            y = lvl(y)
            lvl.step()
        return x + self.dropout(y)

    @torch.no_grad()
    def apply_inner_updates(self, inner_lr: float, inner_scale_xtx: float):
        self.fast.Wq.apply_nlgd_update(inner_lr, inner_scale_xtx)
        self.fast.Wk.apply_nlgd_update(inner_lr, inner_scale_xtx)
        self.fast.Wv.apply_nlgd_update(inner_lr, inner_scale_xtx)
        for lvl in self.cms:
            lvl.apply_inner_updates(inner_lr=0.0, inner_scale_xtx=inner_scale_xtx)  # default no inner updates for CMS

class HOPELM(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, d_kv, max_seq_len, dropout, cms_levels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks = nn.ModuleList([HOPEBlock(d_model, d_kv, cms_levels, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def reset_fast_states(self, B, device):
        for blk in self.blocks:
            blk.fast.reset_fast_state(B, device)

    def forward(self, idx):
        """
        idx: [B, T] token ids
        """
        B, T = idx.size()
        x = self.embed(idx) + self.pos[:, :T, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def apply_inner_updates(self, inner_lr: float, inner_scale_xtx: float):
        for blk in self.blocks:
            blk.apply_inner_updates(inner_lr, inner_scale_xtx)
