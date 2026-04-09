import math
import torch
import torch.nn as nn


def _sample_categorical(categorical_probs):
    '''Gumbel-max categorical sampling (bd3lms diffusion.py).

    Equivalent to torch.multinomial but differentiable and GPU-friendly.
    categorical_probs: (..., vocab_size) non-negative probabilities (not log).
    '''
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


class LogLinearNoise:
    # Built such that 1 - 1/e^(n(t)) interpolates between 0 and ~1 when t varies from 0 to 1. 
    # Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t.
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.total_noise(torch.tensor(0.0)) + self.eps

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def forward(self, t):
        return -1 / t, t #
    
    
class TimestepEmbedder(nn.Module):
    '''Sinusoidal sigma → dense embedding (from bd3lms/models/dit.py).

    Used for optional time conditioning: sigma is mapped to a d_model vector
    that is added to decoder hidden states after the input embedding.
    '''
    def __init__(self, d_model, freq_embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.freq_embed_dim = freq_embed_dim

    @staticmethod
    def sinusoidal_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=t.dtype, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, sigma):
        if sigma.dim() > 1: sigma = sigma.squeeze(-1) # (B,)
        freq_emb = self.sinusoidal_embedding(sigma, self.freq_embed_dim)
        return self.mlp(freq_emb) # (B, d_model)