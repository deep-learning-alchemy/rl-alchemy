"""Policy-gradient diagnostics. All functions are side-effect free."""
import torch
from torch.nn import functional as F


def flat_norm(tensors):
    return torch.sqrt(sum((t.detach().float() ** 2).sum() for t in tensors)).item()


def entropy(logits):
    logp = F.log_softmax(logits, -1)
    return -(logp.exp() * logp).sum(-1)


def kl(logits_p, logits_q):
    """KL(p || q) per state for categorical distributions given by logits."""
    logp, logq = F.log_softmax(logits_p, -1), F.log_softmax(logits_q, -1)
    return (logp.exp() * (logp - logq)).sum(-1)


def srank(features, delta=0.01):
    """Effective rank (Kumar et al., 2021) of a feature matrix."""
    s = torch.linalg.svdvals(features)
    cumulative = torch.cumsum(s, 0) / s.sum().clamp_min(1e-12)
    return int((cumulative < 1 - delta).sum().item()) + 1


def explained_variance(prediction, target):
    var = target.var()
    return float("nan") if var == 0 else (1 - (target - prediction).var() / var).item()


def gradient_noise_scale(per_example_grads):
    """McCandlish et al. (2018) simple noise scale B = tr(Sigma) / |G|^2.

    per_example_grads: (n, P), one flattened gradient per example (here: per
    episode). Uses the unbiased estimate |G|^2 = |g_bar|^2 - tr(Sigma)/n.
    Returns (B_noise, tr_sigma, g_norm_sq); B_noise is nan when |G|^2 <= 0.
    """
    n = per_example_grads.shape[0]
    g_bar = per_example_grads.mean(0)
    tr_sigma = ((per_example_grads - g_bar) ** 2).sum(1).sum() / (n - 1)
    g_sq = (g_bar ** 2).sum() - tr_sigma / n
    b = (tr_sigma / g_sq).item() if g_sq > 0 else float("nan")
    return b, tr_sigma.item(), g_sq.item()
