"""Predictive-uncertainty utilities: scale head, Gaussian / Student-t likelihoods and evaluation metrics."""
import math
import numpy as np
import torch
from torch import nn
from scipy import stats


def inv_softplus(y):
    return math.log(math.expm1(y))


class ScaleHead(nn.Module):
    """sigma^2 = Softplus(MLP(H)) + eps. For dist='studentt' a global degree of freedom nu = 2 + softplus(rho)
    (> 2, finite variance) is learned as well. Initialised to sigma = 1 (= std of the z-scored label)."""

    def __init__(self, d_in, d_hidden=64, dist='studentt', eps=1e-4, nu_init=5.0):
        super().__init__()
        assert dist in ('gaussian', 'studentt')
        self.dist, self.eps = dist, eps
        self.mlp = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, inv_softplus(1.0 - eps))
        if dist == 'studentt':
            self.nu_raw = nn.Parameter(torch.tensor(inv_softplus(nu_init - 2.0)))

    def forward(self, h):
        var = nn.functional.softplus(self.mlp(h).squeeze(-1)) + self.eps
        out = {'scale': var.sqrt()}
        if self.dist == 'studentt':
            out['nu'] = (2.0 + nn.functional.softplus(self.nu_raw)).expand_as(out['scale'])
        return out


def nll_torch(y, mu, scale, nu=None):
    """Per-sample negative log-likelihood of y under N(mu, scale^2) or StudentT(nu, mu, scale)."""
    z = (y - mu) / scale
    if nu is None:
        return 0.5 * math.log(2 * math.pi) + torch.log(scale) + 0.5 * z ** 2
    return (-torch.lgamma((nu + 1) / 2) + torch.lgamma(nu / 2) + 0.5 * torch.log(nu * math.pi)
            + torch.log(scale) + (nu + 1) / 2 * torch.log1p(z ** 2 / nu))


def nll_np(y, mu, scale, nu=None):
    if nu is None:
        return -stats.norm.logpdf(y, loc=mu, scale=scale)
    return -stats.t.logpdf(y, df=nu, loc=mu, scale=scale)


def half_width(scale, nu, level):
    q = stats.norm.ppf(0.5 + level / 2) if nu is None else stats.t.ppf(0.5 + level / 2, df=nu)
    return q * scale


def std_of(scale, nu):
    return scale if nu is None else scale * np.sqrt(nu / (nu - 2))


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z (rank-based partial correlation)."""
    rx, ry, rz = (stats.rankdata(a) for a in (x, y, z))
    r = np.corrcoef(np.vstack([rx, ry, rz]))
    rxy, rxz, ryz = r[0, 1], r[0, 2], r[1, 2]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def prob_metrics(mu, scale, y, nu=None, hv=None, day=None):
    """Evaluation of probabilistic forecasts on z-scored labels. nu: None (Gaussian), scalar or array."""
    v = ~np.isnan(y) & np.isfinite(scale)
    if hv is not None:
        v &= np.isfinite(hv)
    mu, scale, y = mu[v], scale[v], y[v]
    nu = None if nu is None else (np.broadcast_to(nu, v.shape)[v] if np.ndim(nu) else nu)
    err = np.abs(y - mu)
    sd = std_of(scale, nu)
    res = {'NLL': float(np.mean(nll_np(y, mu, scale, nu))), 'sd_mean': float(sd.mean()), 'sd_cv': float(sd.std() / sd.mean())}
    if nu is not None:
        res['nu'] = float(np.mean(nu))
    calib = {}
    for lv in LEVELS:
        hw = half_width(scale, nu, lv)
        calib[lv] = float(np.mean(err <= hw))
        res[f'PICP{int(lv * 100)}'] = calib[lv]
        res[f'MPIW{int(lv * 100)}'] = float(np.mean(2 * hw))
    res['calibration'] = calib
    res['ECE'] = float(np.mean([abs(calib[lv] - lv) for lv in LEVELS]))
    res['spearman_sd_abserr'] = float(stats.spearmanr(sd, err)[0])
    if hv is not None:
        h = hv[v]
        res['spearman_hv_abserr'] = float(stats.spearmanr(h, err)[0])
        res['spearman_sd_hv'] = float(stats.spearmanr(sd, h)[0])
        res['partial_sd_abserr_given_hv'] = float(partial_spearman(sd, err, h))
        # incremental explanatory power for |error| beyond historical volatility (OLS R^2)
        X1 = np.c_[np.ones_like(h), h]
        X2 = np.c_[X1, sd]
        r2 = lambda X: 1 - np.sum((err - X @ np.linalg.lstsq(X, err, rcond=None)[0]) ** 2) / np.sum((err - err.mean()) ** 2)
        res['R2_hv'], res['R2_hv_sd'] = float(r2(X1)), float(r2(X2))
    if day is not None:
        d = day[v]
        rs = [stats.spearmanr(sd[d == k], err[d == k])[0] for k in np.unique(d) if (d == k).sum() > 20]
        res['daily_spearman_sd_abserr'] = float(np.nanmean(rs))
    return res
