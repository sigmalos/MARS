"""MARS: Market-Aware Relational Stock Transformer (relation-enhanced MASTER).

Modules
  master       MASTER backbone + multi-relation GAT, relation attention, gated fusion, dynamic correlation graphs
  base_model   training / evaluation loop (day batches, stopping protocols, turnover penalty, trajectories)
  relations    attach pre-computed relation arrays / return matrix to qlib TSDataSampler
  uncertainty  scale head, Gaussian / Student-t likelihoods, probabilistic metrics
  backtest     Qlib TopkDropout backtest wrapper and vectorised top-k backtest
"""
