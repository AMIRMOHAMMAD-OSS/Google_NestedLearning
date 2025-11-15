import torch
import torch.nn as nn

class NLLinear(nn.Linear):
    """
    A Linear layer that caches X^T X during forward and provides an
    `apply_nlgd_update(lr, scale_xtx)` method implementing:
        W_{t+1} = W_t (I - scale * X^T X) - lr * grad_W
    applied per batch step (no backprop through the update).
    This realizes the paper's L2-regression-step variant of GD (Eq. 28–29).  :contentReference[oaicite:13]{index=13}
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("_xtx", torch.zeros(in_features, in_features), persistent=False)

    def forward(self, x):
        # Cache X^T X (averaged over batch & time if batched 3D)
        if x.dim() == 3:
            B, T, D = x.shape
            x2d = x.reshape(B*T, D)
        else:
            x2d = x
        with torch.no_grad():
            self._xtx = x2d.T @ x2d / max(1, x2d.size(0))
        return super().forward(x)

    @torch.no_grad()
    def apply_nlgd_update(self, lr: float, scale_xtx: float = 1.0):
        if self.weight.grad is None:
            return
        W = self.weight.data                                    # [out, in]
        in_dim = W.size(1)
        # Right multiply by (I - scale*X^T X)
        I = torch.eye(in_dim, device=W.device, dtype=W.dtype)
        W.mul_(1.0)  # no-op for clarity
        W @= (I - scale_xtx * self._xtx)
        # Subtract (outer) gradient step
        W.add_( -lr * self.weight.grad.data )
        if self.bias is not None and self.bias.grad is not None:
            self.bias.data.add_( -lr * self.bias.grad.data )
        # Zero grads on inner-updated params to avoid double stepping by outer optimizer
        self.weight.grad = None
        if self.bias is not None:
            self.bias.grad = None
