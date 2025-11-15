import torch
import torch.nn as nn
import torch.nn.functional as F
from nl.modules.nl_linear import NLLinear

class CMSBlock(nn.Module):
    """
    One level of the Continuum Memory System (CMS). Updated every `update_every` steps.
    Per NL, each level has its own context (update) frequency.  :contentReference[oaicite:14]{index=14}
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float, update_every: int):
        super().__init__()
        self.update_every = int(update_every)
        self.fc1 = NLLinear(d_model, d_ff)
        self.fc2 = NLLinear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self._step = 0

    def forward(self, x):
        return x + self.dropout(self.fc2(self.act(self.fc1(x))))

    def step(self):
        self._step += 1

    def should_update_now(self) -> bool:
        return (self._step % self.update_every) == 0

    @torch.no_grad()
    def apply_inner_updates(self, inner_lr: float, inner_scale_xtx: float):
        # Optionally allow inner updates for CMS too (usually 0 inner_lr so it's a no-op)
        self.fc1.apply_nlgd_update(inner_lr, inner_scale_xtx)
        self.fc2.apply_nlgd_update(inner_lr, inner_scale_xtx)
