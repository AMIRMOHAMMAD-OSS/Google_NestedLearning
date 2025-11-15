import torch
import torch.nn as nn
import torch.nn.functional as F

class FastKV(nn.Module):
    """
    Matrix-valued associative memory:
        M_{t+1} = M_t + v_t k_t^T
        y_t     = M_t q_t
    Implements single-head linear attention over a sequence given Q/K/V projections.
    """
    def __init__(self, d_model: int, d_kv: int):
        super().__init__()
        self.d_model = d_model
        self.d_kv = d_kv
        self.Wq = nn.Linear(d_model, d_kv, bias=False)
        self.Wk = nn.Linear(d_model, d_kv, bias=False)
        self.Wv = nn.Linear(d_model, d_kv, bias=False)
        self.Wo = nn.Linear(d_kv, d_model, bias=False)

        # ✅ Register the buffer only ONCE here
        self.register_buffer("_M", None, persistent=False)

    def reset_fast_state(self, batch_size: int = 0, device=None):
        # ✅ Just ASSIGN to the existing buffer
        if batch_size > 0:
            self._M = torch.zeros(batch_size, self.d_kv, self.d_kv,
                                  device=device, dtype=self.Wq.weight.dtype)
        else:
            self._M = None

    @torch.no_grad()
    def inner_zero_fast(self):
        if isinstance(self._M, torch.Tensor):
            self._M.zero_()

    def forward(self, x):
        """
        x: [B, T, d_model]
        returns: y: [B, T, d_model]
        """
        B, T, D = x.size()
        if (not isinstance(self._M, torch.Tensor)) or (self._M.size(0) != B):
            # use x.device so it matches the input
            self.reset_fast_state(batch_size=B, device=x.device)

        M = self._M
        y_out = []
        for t in range(T):
            xt = x[:, t, :]                                        # [B, D]
            qt = self.Wq(xt)                                       # [B, d_kv]
            kt = self.Wk(xt)
            vt = self.Wv(xt)                                       # value dim == d_kv
            yt = torch.bmm(M, qt.unsqueeze(-1)).squeeze(-1)        # [B, d_kv]
            y_out.append(self.Wo(yt))
            # Hebbian-like update (unnormalized linear attention)
            M = M + torch.bmm(vt.unsqueeze(-1), kt.unsqueeze(1))

        self._M = M
        return torch.stack(y_out, dim=1)
