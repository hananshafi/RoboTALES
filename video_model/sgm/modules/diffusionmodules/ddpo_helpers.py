# --- DDPO helpers ------------------------------------------------------------
import torch


class EMA:
    def __init__(self, beta=0.9): self.beta, self.v, self.inited = beta, 0.0, False
    def update(self, x: torch.Tensor):
        m = x.mean().detach()
        if not self.inited: self.v, self.inited = m, True
        else: self.v = self.beta * self.v + (1 - self.beta) * m
        return self.v

@torch.no_grad()
def decode_latents_to_images(network, latents: torch.Tensor) -> torch.Tensor:
    """
    Latents -> pixel space in [0,1]. Adapt names if your model differs.
    """
    if hasattr(network, "first_stage_model"):  # common in SVD/SGM
        imgs = network.first_stage_model.decode(latents / getattr(network, "scale_factor", 1.0))
    elif hasattr(network, "autoencoder") and hasattr(network.autoencoder, "decode"):
        imgs = network.autoencoder.decode(latents)
    else:
        raise RuntimeError("Can't find a decoder to turn latents into images")
    imgs = (imgs.clamp(-1, 1) + 1) / 2.0
    return imgs

def gaussian_log_prob(x, mean, var_eps):
    # var_eps is scalar or tensor variance (σ^2) per-sample; clamp for stability
    var = torch.clamp(var_eps, 1e-12)
    logp = -0.5 * ( (x - mean).pow(2) / var + torch.log(2 * torch.pi * var) )
    # sum over all non-batch dims
    return logp.flatten(1).sum(-1)

def get_ancestral_sigma_up(sigma_t, sigma_next, eta=1.0):
    """
    Kaiser/Crowson ancestral step: sigma_down = sigma_next, sigma_up^2 chosen
    so total variance matches the discretized SDE. For eta=1 (ancestral),
    a practical choice is:
        sigma_up = eta * ((sigma_next**2) * (1 - (sigma_next / sigma_t)**2)).sqrt()
    See k-diffusion / diffusers Euler-Ancestral scheduler. 
    """
    r = torch.clamp(sigma_next / torch.clamp(sigma_t, 1e-12), max=1.0)
    sig_up2 = (sigma_next**2) * torch.clamp(1 - r**2, min=0.0)
    return sig_up2.sqrt()

def euler_ancestral_step(x_t, sigma_t, sigma_next, x0_pred, noise=None, eta=1.0):
    """
    One Euler-Ancestral step:
        d = (x_t - x0_pred)/sigma_t
        x_mean = x_t + (sigma_next - sigma_t) * d
        x_{t-1} ~ N(x_mean, sigma_up^2 I) with sigma_up from _get_ancestral_sigma_up
    Returns: x_prev, mean, var (for log-prob)
    """
    d = (x_t - x0_pred) / torch.clamp(sigma_t, 1e-12)
    x_mean = x_t + (sigma_next - sigma_t) * d
    sigma_up = get_ancestral_sigma_up(sigma_t, sigma_next, eta=eta)
    if noise is None:
        noise = torch.randn_like(x_t)
    x_prev = x_mean + sigma_up * noise
    var = sigma_up.pow(2)
    return x_prev, x_mean, var
