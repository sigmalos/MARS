import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import os
import numpy as np
import pandas as pd
from scipy import stats

from base_model import SequenceModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x, tau=None):  # tau: [N, nhead] multiplicative logit scale (volatility-aware)
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            logits = torch.matmul(qh, kh.transpose(1, 2)) / self.temperature  # [T, N, N]
            if tau is not None:
                logits = logits * tau[:, i].view(1, -1, 1)
            atten_ave_matrixh = torch.softmax(logits, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x, tau=None):  # tau: [N, nhead]
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            logits = torch.matmul(qh, kh.transpose(1, 2))  # [N, T, T]
            if tau is not None:
                logits = logits * tau[:, i].view(-1, 1, 1)
            atten_ave_matrixh = torch.softmax(logits, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output =d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output


# std of index daily returns over 5/10/20/30/60 days for CSI300, CSI500, CSI800 (positions inside the 63-dim
# market-information block, see data/csi_market_information.csv)
VOL_FEATURE_IDX = [2, 6, 10, 14, 18, 23, 27, 31, 35, 39, 44, 48, 52, 56, 60]


class VolatilityEncoder(nn.Module):
    """Direction D: encodes the market volatility state and outputs one temperature multiplier per attention
    head, tau in (0, 2). tau > 1 sharpens the attention (focus on few stocks / days), tau < 1 smooths it."""
    def __init__(self, d_input, n_out, d_hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_input, d_hidden), nn.ReLU(), nn.Linear(d_hidden, n_out))
        nn.init.zeros_(self.net[-1].weight)  # tau == 1 at init -> identical to the original attention
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, vol):
        return 2 * torch.sigmoid(self.net(vol))


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MultiScaleTemporalConv(nn.Module):
    """Direction A: parallel causal depthwise Conv1D branches with different kernel sizes / dilations
    along the time axis, fused by concatenation + linear projection and added back residually."""
    def __init__(self, d_model, kernels=(1, 3, 5), dilations=None, dropout=0.0):
        super().__init__()
        dilations = dilations or [1] * len(kernels)
        self.pads = [(k - 1) * d for k, d in zip(kernels, dilations)]
        self.branches = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, dilation=d, groups=d_model)
            for k, d in zip(kernels, dilations)
        ])
        self.act = nn.GELU()
        self.fuse = nn.Linear(d_model * len(kernels), d_model)
        nn.init.zeros_(self.fuse.weight)  # start as identity mapping of the original model
        nn.init.zeros_(self.fuse.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [N, T, D]
        h = x.transpose(1, 2)  # [N, D, T]
        outs = [self.act(conv(nn.functional.pad(h, (p, 0)))) for conv, p in zip(self.branches, self.pads)]
        h = torch.cat(outs, dim=1).transpose(1, 2)  # [N, T, D*K]
        return x + self.dropout(self.fuse(h))


class DynamicMarketGate(nn.Module):
    """Direction B: market-state encoder (GRU over the T-step market-information sequence) that produces
    time-step specific gate logits. They are added to the original static gate logits, so the feature
    weights change with the evolving market state instead of being a single vector for the whole window."""
    def __init__(self, static_gate, d_input, d_output, d_hidden=64, beta=1.0):
        super().__init__()
        self.static_gate = static_gate
        self.encoder = nn.GRU(d_input, d_hidden, batch_first=True)
        self.dyn = nn.Linear(d_hidden, d_output)
        nn.init.zeros_(self.dyn.weight)  # starts identical to the original gate
        nn.init.zeros_(self.dyn.bias)
        self.d_output = d_output
        self.t = beta

    def forward(self, market_seq):  # [N, T, F']
        # Market information is (almost always) identical across stocks of a day; it differs only for stocks
        # with suspended days whose look-back window covers other dates, so we encode per stock.
        h, _ = self.encoder(market_seq)                                           # [N, T, H]
        static_logit = self.static_gate.trans(market_seq[:, -1, :]).unsqueeze(1)  # [N, 1, D]
        logit = static_logit + self.dyn(h)                                        # [N, T, D]
        return self.d_output * torch.softmax(logit / self.t, dim=-1)


class GraphAttention(nn.Module):
    """Direction C: dense multi-head GAT (Velickovic et al.) over a stock relation graph, applied at every
    time step. adj[i, j] = True means stock j is a neighbour of stock i (self-loops included)."""
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.nhead = nhead
        self.dh = d_model // nhead
        self.norm = LayerNorm(d_model, eps=1e-5)
        self.W = nn.Linear(d_model, self.dh * nhead, bias=False)
        self.a_src = nn.Parameter(torch.empty(nhead, self.dh))
        self.a_dst = nn.Parameter(torch.empty(nhead, self.dh))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky = nn.LeakyReLU(0.2)
        self.attn_dropout = Dropout(dropout)
        self.out = nn.Linear(self.dh * nhead, d_model)

    def forward(self, x, adj):  # x: [N, T, D], adj: [N, N] bool
        N, T, _ = x.shape
        h = self.W(self.norm(x)).view(N, T, self.nhead, self.dh).permute(1, 2, 0, 3)  # [T, H, N, dh]
        e_src = (h * self.a_src[None, :, None, :]).sum(-1)                            # [T, H, N]
        e_dst = (h * self.a_dst[None, :, None, :]).sum(-1)
        e = self.leaky(e_src.unsqueeze(-1) + e_dst.unsqueeze(-2))                     # [T, H, N, N]
        e = e.masked_fill(~adj, float('-inf'))
        att = self.attn_dropout(torch.softmax(e, dim=-1))
        out = torch.matmul(att, h)                                                    # [T, H, N, dh]
        out = out.permute(2, 0, 1, 3).reshape(N, T, -1)
        return torch.nn.functional.elu(self.out(out))


class GatedFusion(nn.Module):
    """Element-wise gate deciding how much graph information is added to the S-Attention output."""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # sigmoid(-2) ~ 0.12: start close to the original model

    def forward(self, s_out, g_out):
        return s_out + torch.sigmoid(self.gate(torch.cat([s_out, g_out], dim=-1))) * g_out


class MASTER(nn.Module):
    """MASTER with switchable improvement modules:
    A ms_kernels   multi-scale temporal Conv1D      B dyn_gate     dynamic market-state feature gate
    C use_gat      industry-graph GAT               D vol_attn     volatility-aware attention temperature
    E uncertainty  mean + log-variance output"""
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta,
                 ms_kernels=(), ms_dilations=None, ms_dropout=0.0, dyn_gate=False, dyn_gate_hidden=64,
                 use_gat=False, gat_nhead=2, gat_dropout=0.1, vol_attn=False, vol_hidden=32, uncertainty=False):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )
        # improvement modules, created after the original layers (identical init of shared parameters per seed)
        self.use_ms = bool(ms_kernels)
        if self.use_ms:
            self.multiscale = MultiScaleTemporalConv(d_model, ms_kernels, ms_dilations, ms_dropout)
        self.use_dyn_gate = dyn_gate
        if dyn_gate:
            self.dyn_gate = DynamicMarketGate(self.feature_gate, self.d_gate_input, d_feat, dyn_gate_hidden, beta)
        self.use_gat = use_gat
        if use_gat:
            self.gat = GraphAttention(d_model, gat_nhead, gat_dropout)
            self.gat_fusion = GatedFusion(d_model)
        self.use_vol = vol_attn
        self.t_nhead = t_nhead
        if vol_attn:
            self.vol_idx = [gate_input_start_index + i for i in VOL_FEATURE_IDX]
            self.vol_encoder = VolatilityEncoder(len(self.vol_idx), t_nhead + s_nhead, vol_hidden)
        self.uncertainty = uncertainty
        if uncertainty:
            self.logvar_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x, adj=None):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        market = x[:, :, self.gate_input_start_index:self.gate_input_end_index]
        if self.use_dyn_gate:                                         # B
            src = src * self.dyn_gate(market)
        else:
            src = src * torch.unsqueeze(self.feature_gate(market[:, -1, :]), dim=1)

        tau_t = tau_s = None
        if self.use_vol:                                              # D
            tau = self.vol_encoder(x[:, -1, self.vol_idx])
            tau_t, tau_s = tau[:, :self.t_nhead], tau[:, self.t_nhead:]

        h = self.layers[1](self.layers[0](src))
        if self.use_ms:                                               # A
            h = self.multiscale(h)
        h = self.layers[2](h, tau_t)
        s_out = self.layers[3](h, tau_s)
        if self.use_gat and adj is not None:                          # C
            s_out = self.gat_fusion(s_out, self.gat(h, adj))
        z = self.layers[4](s_out)
        mu = self.layers[5](z).squeeze(-1)
        if not self.uncertainty:
            return mu
        raw = self.logvar_head(z).squeeze(-1)                         # E
        return mu, 6.0 * torch.tanh(raw / 6.0)


MODULE_KWARGS = ['ms_kernels', 'ms_dilations', 'ms_dropout', 'dyn_gate', 'dyn_gate_hidden', 'use_gat', 'gat_nhead',
                 'gat_dropout', 'vol_attn', 'vol_hidden', 'uncertainty']


class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, nll_beta=0.0, industry_file=None, **kwargs,
    ):
        self.module_kwargs = {k: kwargs.pop(k) for k in MODULE_KWARGS if k in kwargs}
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        self.uncertainty = self.module_kwargs.get('uncertainty', False)
        self.use_gat = self.module_kwargs.get('use_gat', False)
        self.nll_beta = nll_beta

        if self.use_gat:
            industry_file = industry_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'stock_industry.csv')
            ind = pd.read_csv(industry_file).fillna({'industry': ''})
            ind = ind[ind['industry'] != '']
            codes = {name: i for i, name in enumerate(sorted(ind['industry'].unique()))}
            self.industry_of = dict(zip(ind['instrument'], ind['industry'].map(codes)))

        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, **self.module_kwargs)
        super(MASTERModel, self).init_model()

    def build_adj(self, inst):
        ids = torch.tensor([self.industry_of.get(c, -1) for c in inst], device=self.device)
        adj = (ids[:, None] == ids[None, :]) & (ids[:, None] >= 0)
        return adj | torch.eye(len(inst), dtype=torch.bool, device=self.device)

    def _forward(self, feature, inst=None):
        adj = self.build_adj(inst) if (self.use_gat and inst is not None) else None
        out = self.model(feature, adj)
        if not self.uncertainty:
            return {'pred': out}
        return {'pred': out[0], 'logvar': out[1]}

    def _train_loss(self, out, label):
        mse = self.loss_fn(out['pred'], label)
        if not self.uncertainty:
            return mse, mse
        mask = ~torch.isnan(label)
        mu, logvar, y = out['pred'][mask], out['logvar'][mask], label[mask]
        nll = 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))
        if self.nll_beta > 0:
            nll = nll * torch.exp(logvar).detach() ** self.nll_beta
        # backprop the (beta-)NLL, but keep the MSE of the mean for the common stop criterion
        return nll.mean(), mse

    def _extra_metrics(self, outs, labels):
        if not self.uncertainty:
            return {}
        res = {}
        mu = np.concatenate([o['pred'].ravel() for o in outs])
        sig = np.exp(0.5 * np.concatenate([o['logvar'].ravel() for o in outs]))
        y = np.concatenate(labels)
        v = ~np.isnan(y)
        mu, sig, y = mu[v], sig[v], y[v]
        err = np.abs(y - mu)
        for name, z in [('68', 1.0), ('90', 1.6449), ('95', 1.96)]:
            res[f'PICP{name}'] = float(np.mean(err <= z * sig))
            res[f'MPIW{name}'] = float(np.mean(2 * z * sig))
            # reference: constant sigma = 1 (std of the z-scored label), i.e. no uncertainty model
            res[f'PICP{name}_const'] = float(np.mean(err <= z))
        res['NLL'] = float(np.mean(0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * (err / sig) ** 2))
        res['NLL_const'] = float(np.mean(0.5 * np.log(2 * np.pi) + 0.5 * err ** 2))
        res['sigma_mean'] = float(sig.mean())
        res['sigma_std'] = float(sig.std())
        res['corr_sigma_abserr'] = float(stats.spearmanr(sig, err)[0])
        # per-day: correlation of sigma with |error| and IC of the confident / unconfident half
        cs, ic_lo, ic_hi = [], [], []
        for o, lab in zip(outs, labels):
            m = ~np.isnan(lab)
            if m.sum() < 20:
                continue
            p, s_, l = o['pred'].ravel()[m], np.exp(0.5 * o['logvar'].ravel()[m]), lab[m]
            cs.append(stats.spearmanr(s_, np.abs(l - p))[0])
            lo = s_ <= np.median(s_)
            ic_lo.append(np.corrcoef(p[lo], l[lo])[0, 1])
            ic_hi.append(np.corrcoef(p[~lo], l[~lo])[0, 1])
        res['daily_corr_sigma_abserr'] = float(np.nanmean(cs))
        res['IC_low_sigma_half'] = float(np.nanmean(ic_lo))
        res['IC_high_sigma_half'] = float(np.nanmean(ic_hi))
        return res
