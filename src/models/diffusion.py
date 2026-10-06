"""Class-conditional DDPM with classifier-free guidance and DDIM sampling.

Forward (noising) process:  x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps
Training: pick random t, add noise, train UNet to predict eps (MSE loss).
Classifier-free guidance (CFG): with prob p_uncond replace the label by the
"null" label, so ONE network learns both p(x|y) and p(x).
At sampling:  eps = (1 + w) * eps(x, t, y) - w * eps(x, t, null)
  w = 0 -> plain conditional model; larger w -> more class-typical samples
  (and, possibly, more memorization - one of our trade-off curves).
DDIM (eta = 0) sampling uses 50 of the 1000 steps deterministically -> ~20x faster.
"""
import math

import torch
import torch.nn.functional as F


def cosine_alphas_cumprod(T, s=0.008):
    """Nichol & Dhariwal cosine schedule: noise is added smoothly, not too fast at the start."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


class Diffusion:
    """pred="v": the network predicts v = sqrt(abar)*eps - sqrt(1-abar)*x0.

    Why v and not eps: at the last steps of the cosine schedule abar ~ 0, so
    recovering x0 from an eps-prediction divides by sqrt(abar) ~ 0 and any
    error explodes. For EDA this blew up the window's mean level (the most
    important feature). v-prediction stays well-posed at every noise level
    (Salimans & Ho 2022; Lin et al. 2024).
    """

    def __init__(self, model, T=1000, p_uncond=0.1, n_classes=2, device="cpu", pred="v"):
        self.model = model
        self.T = T
        self.p_uncond = p_uncond
        self.null = n_classes          # index of the null label
        self.device = device
        self.pred = pred
        self.abar = cosine_alphas_cumprod(T).to(device)

    def loss(self, x0, y):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        eps = torch.randn_like(x0)
        ab = self.abar[t][:, None, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        drop = torch.rand(b, device=x0.device) < self.p_uncond
        y = torch.where(drop, torch.full_like(y, self.null), y)
        target = eps if self.pred == "eps" else ab.sqrt() * eps - (1 - ab).sqrt() * x0
        return F.mse_loss(self.model(xt, t, y), target)

    def _x0_eps(self, out, x, ab):
        """Convert network output to (x0_hat, eps_hat)."""
        if self.pred == "eps":
            return (x - (1 - ab).sqrt() * out) / ab.sqrt(), out
        x0 = ab.sqrt() * x - (1 - ab).sqrt() * out
        eps = (1 - ab).sqrt() * x + ab.sqrt() * out
        return x0, eps

    @torch.no_grad()
    def sample(self, y, length=240, steps=50, guidance=2.0, model=None, generator=None, clip=8.0):
        """DDIM (eta=0) sampling for a batch of labels y [B]."""
        net = model or self.model
        net.eval()
        b = len(y)
        x = torch.randn(b, 1, length, device=self.device, generator=generator)
        ts = torch.linspace(self.T - 1, 0, steps, device=self.device).long()
        null = torch.full_like(y, self.null)
        for i, t in enumerate(ts):
            tt = t.repeat(b)
            if guidance == 0:
                out = net(x, tt, y)
            else:
                o = net(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, null]))
                oc, ou = o.chunk(2)
                out = (1 + guidance) * oc - guidance * ou   # CFG (linear, so valid for eps or v)
            ab = self.abar[t]
            x0, eps = self._x0_eps(out, x, ab)
            x0 = x0.clamp(-clip, clip)
            ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=self.device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x.squeeze(1)

    def generate(self, n_per_class, guidance=2.0, steps=50, seed=0, bs=512, model=None, length=240):
        """Return (X [N, L] numpy, y [N]) with n_per_class[c] samples of class c."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        Xs, ys = [], []
        for c, n in enumerate(n_per_class):
            for i in range(0, n, bs):
                m = min(bs, n - i)
                y = torch.full((m,), c, device=self.device, dtype=torch.long)
                Xs.append(self.sample(y, length, steps, guidance, model, g).cpu())
                ys.append(torch.full((m,), c))
        return torch.cat(Xs).numpy(), torch.cat(ys).numpy()
