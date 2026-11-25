import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import numpy as np

from .base_model import SequenceModel


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

    def forward(self, x):
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

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
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

    def forward(self, x):
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
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
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


class GraphAttention(nn.Module):
    """Dense multi-head GAT (Velickovic et al.) over a stock relation graph, applied at every time step.
    adj[i, j] = True means stock j is a neighbour of stock i (self-loops included).

    edge = 'none'   : unweighted (pre-experiment / formal-experiment model)
           'weight' : attention logit bias  e_ij += b_h * w_ij        (w = correlation of the edge, 0 on self-loops)
           'signed' : e_ij += b_h * |w_ij| + c_h * [w_ij < 0]  and sign-specific value projection: messages of
                      negatively correlated neighbours use W_neg (hedging relation), positive ones W.
    New parameters are zero (b, c) or a copy of W (W_neg), so every edge mode equals the unweighted GAT at init."""
    def __init__(self, d_model, nhead, dropout, edge='none'):
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
        assert edge in ('none', 'weight', 'signed')
        self.edge = edge
        if edge != 'none':
            self.edge_bias = nn.Parameter(torch.zeros(nhead))
        if edge == 'signed':
            self.neg_bias = nn.Parameter(torch.zeros(nhead))
            with torch.random.fork_rng(devices=[]):   # do not consume the global RNG: later modules keep their init
                self.W_neg = nn.Linear(d_model, self.dh * nhead, bias=False)
            with torch.no_grad():
                self.W_neg.weight.copy_(self.W.weight)

    def forward(self, x, adj, w=None):  # x: [N, T, D], adj: [N, N] bool, w: [N, N] edge correlation (0 = none)
        N, T, _ = x.shape
        xn = self.norm(x)
        h = self.W(xn).view(N, T, self.nhead, self.dh).permute(1, 2, 0, 3)            # [T, H, N, dh]
        e_src = (h * self.a_src[None, :, None, :]).sum(-1)                            # [T, H, N]
        e_dst = (h * self.a_dst[None, :, None, :]).sum(-1)
        e = self.leaky(e_src.unsqueeze(-1) + e_dst.unsqueeze(-2))                     # [T, H, N, N]
        if self.edge == 'weight':
            e = e + self.edge_bias.view(1, -1, 1, 1) * w
        elif self.edge == 'signed':
            neg = (w < 0).float()
            e = e + self.edge_bias.view(1, -1, 1, 1) * w.abs() + self.neg_bias.view(1, -1, 1, 1) * neg
        e = e.masked_fill(~adj, float('-inf'))
        att = self.attn_dropout(torch.softmax(e, dim=-1))
        if self.edge == 'signed':
            hn = self.W_neg(xn).view(N, T, self.nhead, self.dh).permute(1, 2, 0, 3)
            out = torch.matmul(att * (1 - neg), h) + torch.matmul(att * neg, hn)
        else:
            out = torch.matmul(att, h)                                                # [T, H, N, dh]
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


class RelationAttention(nn.Module):
    """Relation-level (semantic) attention: alpha_r = softmax_r( q^T tanh(W H_r + b) ), computed per stock and
    time step; H_graph = sum_r alpha_r H_r. q is zero-initialised, i.e. all relations start with equal weight."""
    def __init__(self, d_model, d_hidden=64):
        super().__init__()
        self.proj = nn.Linear(d_model, d_hidden)
        self.q = nn.Linear(d_hidden, 1, bias=False)
        nn.init.zeros_(self.q.weight)

    def forward(self, hs):
        H = torch.stack(hs, dim=-2)                                     # [N, T, R, D]
        alpha = torch.softmax(self.q(torch.tanh(self.proj(H))).squeeze(-1), dim=-1)  # [N, T, R]
        return (alpha.unsqueeze(-1) * H).sum(dim=-2), alpha


class MultiRelationGAT(nn.Module):
    """One GAT per relation graph (own parameters), fused by relation-level attention."""
    def __init__(self, relations, d_model, nhead, dropout, rel_hidden=64, edge_modes=None):
        super().__init__()
        self.relations = list(relations)
        edge_modes = edge_modes or {}
        self.gats = nn.ModuleDict({r: GraphAttention(d_model, nhead, dropout, edge_modes.get(r, 'none')) for r in self.relations})
        self.rel_attn = RelationAttention(d_model, rel_hidden) if len(self.relations) > 1 else None

    def forward(self, h, adjs, weights=None):
        weights = weights or {}
        hs = [self.gats[r](h, adjs[r], weights.get(r)) for r in self.relations]
        if self.rel_attn is None:
            return hs[0], None
        return self.rel_attn(hs)


# relation name -> (kind, array name in the relation files built by scripts/build_relations.py)
RELATIONS = {
    'ind':  ('cat', 'csrc'),       # CSRC industry, point-in-time (baostock monthly snapshots)
    'corr': ('knn', 'corr10'),     # top-10 Pearson correlation of the previous 60 daily returns
    'dcorr': ('dyn', None),        # correlation graph computed on the fly (window / sign mode: graph_cfg)
    'fine': ('cat', 'sw2'),        # Shenwan level-2, point-in-time (cninfo change history)
    'sw1':  ('cat', 'sw1'),        # Shenwan level-1 (granularity study)
    'sw3':  ('cat', 'sw3'),        # Shenwan level-3 (granularity study)
    'snap': ('cat', 'csrc_snap'),  # CSRC industry, 2026-11 snapshot back-filled (= pre-experiment graph)
    'rand': ('cat', 'csrc_rand'),  # control: point-in-time CSRC labels of a randomly permuted stock
}
CANONICAL_ORDER = list(RELATIONS)


class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta,
                 relations=(), gat_nhead=2, gat_dropout=0.1, rel_hidden=64, joint_unc=None, unc_hidden=64, edge_modes=None):
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
        # new modules are created after the original layers -> identical init of shared parameters per seed
        self.relations = [r for r in CANONICAL_ORDER if r in relations]
        if self.relations:
            self.graph = MultiRelationGAT(self.relations, d_model, gat_nhead, gat_dropout, rel_hidden, edge_modes)
            self.gat_fusion = GatedFusion(d_model)
        self.unc_head = None
        if joint_unc:
            from .uncertainty import ScaleHead
            self.unc_head = ScaleHead(d_model, unc_hidden, dist=joint_unc)

    def forward(self, x, adjs=None, return_repr=False, weights=None):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        h = self.layers[2](self.layers[1](self.layers[0](src)))   # feature layer, PE, intra-stock T-Attention
        s = self.layers[3](h)                                      # inter-stock S-Attention
        out = {}
        if self.relations and adjs is not None:
            g, alpha = self.graph(h, adjs, weights)                # multi-relation GAT on the same input
            s = self.gat_fusion(s, g)                              # H_SA + gamma * H_graph
            if alpha is not None:
                out['rel_alpha'] = alpha.mean(dim=1)               # [N, R], averaged over time steps
        z = self.layers[4](s)                                      # temporal aggregation -> [N, D]
        out['pred'] = self.layers[5](z).squeeze(-1)
        if self.unc_head is not None:
            out.update(self.unc_head(z))
        if return_repr:
            out['repr'] = z
        return out


class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, relations=(), gat_nhead=2, gat_dropout=0.1, rel_hidden=64,
            joint_unc=None, unc_hidden=64, corr_window=60, corr_mode='pos', corr_k=10, corr_kneg=5, corr_edge='none',
            ind_edge='none', edge_window=None, **kwargs,
    ):
        super(MASTERModel, self).__init__(**kwargs)
        # dynamic graph configuration (relation 'dcorr' and edge-weighted 'ind'):
        #   corr_window : look-back window W of the correlation graph (returns strictly before day t)
        #   corr_mode   : 'pos'    top-k raw correlation (as the stored 60-day graph)
        #                 'signed' top-k most positive + top-k_neg most negative market-residual correlation
        #   corr_edge   : GAT edge mode of 'dcorr' ('none' / 'weight' / 'signed')
        #   ind_edge    : GAT edge mode of 'ind'; edge weight = market-residual correlation over edge_window
        self.corr_window, self.corr_mode, self.corr_k, self.corr_kneg = corr_window, corr_mode, corr_k, corr_kneg
        self.corr_edge, self.ind_edge = corr_edge, ind_edge
        self.edge_window = edge_window or corr_window
        assert corr_mode in ('pos', 'signed') and not (corr_mode == 'signed' and corr_edge != 'signed')
        self._ret_gpu = None
        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        unknown = set(relations) - set(RELATIONS)
        assert not unknown, f'unknown relations {unknown}'
        self.relations = [r for r in CANONICAL_ORDER if r in relations]
        self.gat_nhead, self.gat_dropout, self.rel_hidden = gat_nhead, gat_dropout, rel_hidden
        self.joint_unc, self.unc_hidden = joint_unc, unc_hidden

        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta,
                                   relations=self.relations, gat_nhead=self.gat_nhead, gat_dropout=self.gat_dropout,
                                   rel_hidden=self.rel_hidden, joint_unc=self.joint_unc, unc_hidden=self.unc_hidden,
                                   edge_modes={'dcorr': self.corr_edge, 'ind': self.ind_edge})
        super(MASTERModel, self).init_model()

    def day_corr(self, info, W, demean):
        """Pearson correlation of the day's stocks over the W returns strictly before day t ([n, n], NaN = invalid).
        Same estimator as scripts/build_relations.py: per-stock mean / std over its valid returns, missing -> 0
        after centring, >= ceil(2W/3) valid returns required. demean: subtract the equal-weighted cross-sectional
        mean return of the day's universe first (market-residual correlation)."""
        if self._ret_gpu is None:
            self._ret_gpu = torch.from_numpy(info['ret']).to(self.device)
        ci = info['rel']['cal_idx']
        assert ci[0] == ci[-1], 'one trading day per batch'
        cols = torch.from_numpy(info['rel']['inst_idx'].astype(np.int64)).to(self.device)
        X = self._ret_gpu[int(ci[0]) - W:int(ci[0])][:, cols].double()          # [W, n]
        m = ~torch.isnan(X)
        if demean:
            rm = torch.where(m, X, torch.zeros_like(X)).sum(1) / m.sum(1).clamp(min=1)
            X = X - rm[:, None]
        cnt = m.sum(0)
        mean = torch.where(m, X, torch.zeros_like(X)).sum(0) / cnt.clamp(min=1)
        Xc = torch.where(m, X - mean, torch.zeros_like(X))
        sd = torch.sqrt((Xc ** 2).sum(0) / (cnt - 1).clamp(min=1))
        valid = (cnt >= int(np.ceil(2 * W / 3))) & (sd > 0)
        Z = torch.where(valid[None, :], Xc / torch.where(sd > 0, sd, torch.ones_like(sd)), torch.zeros_like(Xc))
        C = (Z.T @ Z / (W - 1)).float()
        C[~valid, :] = float('nan')
        C[:, ~valid] = float('nan')
        C.fill_diagonal_(float('nan'))
        return C

    @staticmethod
    def _knn(C, k, largest):
        """adjacency [n, n]: row i -> its k largest (or smallest) finite entries."""
        n = C.shape[0]
        fill = float('-inf') if largest else float('inf')
        Cm = torch.where(torch.isfinite(C), C, torch.full_like(C, fill))
        nvalid = torch.isfinite(C).any(1)
        kk = min(k, int(nvalid.sum()) - 1)
        adj = torch.zeros(n, n, dtype=torch.bool, device=C.device)
        if kk > 0:
            top = torch.topk(Cm, kk, dim=1, largest=largest).indices
            rows = torch.arange(n, device=C.device)[:, None].expand(-1, kk)
            ok = nvalid[:, None] & torch.isfinite(torch.gather(C, 1, top))
            adj[rows[ok], top[ok]] = True
        return adj

    def build_adjs(self, info, keep=None):
        adjs, weights = {}, {}
        n = len(info['inst'])
        eye = torch.eye(n, dtype=torch.bool, device=self.device)
        cache = {}

        def corr(W, demean):
            if (W, demean) not in cache:
                cache[(W, demean)] = self.day_corr(info, W, demean)
            return cache[(W, demean)]

        for r in self.relations:
            kind, key = RELATIONS[r]
            w = None
            if kind == 'cat':
                arr = torch.from_numpy(np.ascontiguousarray(info['rel'][key])).to(self.device)
                adj = (arr[:, None] == arr[None, :]) & (arr[:, None] >= 0)
                if r == 'ind' and self.ind_edge != 'none':
                    w = torch.nan_to_num(corr(self.edge_window, True), nan=0.0)
            elif kind == 'knn':  # arr[i] = positions (within the day) of the neighbours of stock i, -1 = none
                arr = torch.from_numpy(np.ascontiguousarray(info['rel'][key])).to(self.device)
                adj = torch.zeros(n, n, dtype=torch.bool, device=self.device)
                rows = torch.arange(n, device=self.device).repeat_interleave(arr.shape[1])
                cols = arr.reshape(-1).long()
                ok = cols >= 0
                adj[rows[ok], cols[ok]] = True
            else:  # 'dyn': correlation graph computed on the fly
                if self.corr_mode == 'pos':
                    C = corr(self.corr_window, False)
                    adj = self._knn(C, self.corr_k, largest=True)
                else:
                    C = corr(self.corr_window, True)
                    adj = self._knn(C, self.corr_k, largest=True) | self._knn(C, self.corr_kneg, largest=False)
                if self.corr_edge != 'none':
                    w = torch.nan_to_num(C, nan=0.0)
            adj = adj | eye
            if w is not None:
                w = torch.where(adj & ~eye, w, torch.zeros_like(w))
            if keep is not None:
                adj = adj[keep][:, keep]
                w = w[keep][:, keep] if w is not None else None
            adjs[r] = adj
            if w is not None:
                weights[r] = w
        return adjs, weights

    def _forward(self, feature, info, keep=None, return_repr=False):
        if not self.relations:
            return self.model(feature, None, return_repr=return_repr)
        adjs, weights = self.build_adjs(info, keep)
        return self.model(feature, adjs, return_repr=return_repr, weights=weights)

    def _train_loss(self, out, label):
        mse = self.loss_fn(out['pred'], label)
        if self.joint_unc is None:
            return mse, mse
        from .uncertainty import nll_torch
        m = ~torch.isnan(label)
        nll = nll_torch(label[m], out['pred'][m], out['scale'][m], out.get('nu', None)[m] if 'nu' in out else None)
        # backprop the NLL jointly, keep the MSE of the mean for the common stop criterion
        return nll.mean(), mse
