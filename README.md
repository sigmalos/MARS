# MARS: Market-Aware Relational Stock Transformer

MARS 是在 [MASTER](https://github.com/SJTU-DMTai/MASTER)（AAAI 2024，*Market-Guided Stock Transformer for Stock Price Forecasting*）基础上做的**关系增强**改进。它保留 MASTER 的 Feature Gate、T-Attention、S-Attention 与 Temporal Attention 主干，并在 S-Attention 旁并联一个**多关系股票图注意力分支**：

- **按时点有效的行业关系图**：证监会行业，来自 baostock 月度快照；申万一/二/三级行业，来自巨潮资讯的行业变更记录；
- **动态收益相关图**：每只股票连向预测日之前 $W$ 个交易日收益最相关的 Top-$k$ 只股票。可选市场残差相关的**带符号边**（对冲关系）；
- 每种关系使用独立的 GAT，经**关系级注意力**融合后，与 S-Attention 的输出**门控相加**。新增模块均零初始化，因此训练初期与 MASTER 等价；
- 可选的**解耦式（两阶段）Student-t 预测不确定性**：冻结均值网络，只训练尺度头，因此不改变排序。

本仓库 fork 自 [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER)，`qlib-update/` 为原仓库内容。

## 主要结果

实验数据为 MASTER 官方开源数据（opensource 版本）：训练集 2008-01—2020-03，测试集 2020-07—2022-12。下表为 10 个随机种子的均值，$\Delta$ 为与同种子 MASTER 的配对差值，括号内为 [提升种子数/10]。

| 模型（官方停止准则：训练 MSE ≤ 0.95） | CSI300 IC | CSI300 RankIC | CSI800 IC | CSI800 RankIC |
| --- | --- | --- | --- | --- |
| MASTER（复现） | 0.0597 | 0.0668 | 0.0469 | 0.0596 |
| MARS：行业 + 相关图 | 0.0671（+0.0074 [10/10]） | 0.0731 | 0.0481 | 0.0606 |
| MARS：行业 + 相关 + 申万二级 | 0.0673（+0.0075 [9/10]） | 0.0727 | 0.0489（+0.0020 [9/10]） | 0.0605 |
| 随机图对照（度分布相同） | 0.0595 | 0.0647 | 0.0445 | 0.0592 |

- 官方停止准则会让参数更多的关系模型提前 5—7 轮停止。若按相同训练轮数比较，多关系模型在两个指数上 10/10 个种子同时提升 IC 与 RankIC：CSI300 为 +0.0070 / +0.0059，CSI800 为 +0.0048 / +0.0028，$p<0.001$。随机图对照没有提升，说明收益来自真实的股票关联。
- **换手约束**：组合仍取 top-30，但每日最多替换 3 只（CSI300）或 5 只（CSI800）股票，即 Qlib TopkDropout 的 `n_drop`。这样可在所有配置上把含费 IR 提高 0.5—1.1。CSI300 上“行业+相关”模型的含费超额年化为 23.5%、IR 1.94，官方设定下的 MASTER 为 14.2%、IR 1.15。
- **两阶段 Student-t 不确定性**：测试集 NLL 由常数方差的 1.426 降至 1.314，平均校准误差 0.011。
- **未能在测试集上稳定提升的方向**：20 日相关窗口、带符号边、在内部验证集上选出的停止规则、训练时的换手惩罚。其中换手惩罚会破坏预测信号。

**推荐配置**：行业 + 相关图（60 日窗口，无符号边），官方停止准则；组合采用 top-30，`n_drop`=3（CSI300）/ 5（CSI800）。汇总后的结果文件见 `results_summary/`。

## 目录

| 路径 | 内容 |
| --- | --- |
| `mars/master.py` | MARS 模型：多关系 GAT（边模式 none / weight / signed）、关系注意力、门控融合、动态相关图、可选联合尺度头 |
| `mars/base_model.py` | 训练与评估：按日向量化取数、官方停止准则与验证集选模、换手惩罚、固定轮数轨迹训练 |
| `mars/relations.py` | 把预计算的关系数组与收益矩阵挂到 Qlib `TSDataSampler` 上 |
| `mars/uncertainty.py` | 尺度头、高斯 / Student-t 似然、概率指标 |
| `mars/backtest.py` | Qlib 官方回测封装（TopkDropout），以及向量化 top-k 回测 |
| `main.py` | 训练 / 评估入口，同时记录官方准则（stop）与验证集最优（best）两种协议 |
| `run_traj.py` | 固定轮数训练并保存每轮预测，支持内部验证切分（`--protocol inner`） |
| `stage2_uncertainty.py` | 两阶段不确定性训练（冻结均值网络，只训练尺度头） |
| `scripts/fetch_industry.py` | 下载按时点的证监会行业（baostock）与行业变更记录（巨潮资讯） |
| `scripts/build_relations.py`、`scripts/build_returns.py` | 构建逐样本的关系数组，以及全局日收益矩阵 |
| `scripts/` 其余文件 | 批量任务（`make_jobs*.py`、`job_queue.py`）、选择与统计（`select_inner.py`、`aggregate.py`、`final_*.py`）、回测与作图 |
| `results_summary/` | 汇总后的实验结果（JSON） |
| `data/`、`model/` | **不纳入版本控制**：数据表与模型文件，获取方式见下文“数据准备” |

## 环境

Python 3.11、PyTorch 2.5、pandas 2.3、numpy 2.x、pyqlib 0.9.7、scipy、matplotlib。下载行业数据另需 `baostock` 与 `cryptography`。

## 数据准备

仓库运行时需要两个本地目录：`data/` 与 `model/`。它们不在版本控制中，可以是普通目录，也可以是指向数据盘的软链接。

- `data/`：
  - `csi_market_information.csv` 取自 MASTER 原仓库的 `data/`；
  - `csrc_pit.csv`、`sw_pit.csv`、`relation_stats.json` 由下面第 3 步的 `build_relations.py --data_out data` 生成；
  - `stock_industry.csv` 由 `fetch_industry.py snapshot` 生成。
- `model/`：MASTER 官方 checkpoint（`csi{300,800}_{original,opensource}_0.pkl`）与 `performance.xlsx`，取自 MASTER 原仓库的 `model/`，仅用于复现官方结果。

1. **MASTER 数据**：从 MASTER 原 README 中的链接下载 opensource 版本数据，例如放到 `/data/dataset/MASTER/opensource/{csi300,csi800}_dl_{train,valid,test}.pkl`。
2. **Qlib 行情**：下载 [chenditc/investment_data](https://github.com/chenditc/investment_data/releases) 发布的 `qlib_bin.tar.gz` 并解压，用于计算相关图、历史波动率和回测。
3. **行业与关系数据**：
   ```bash
   python scripts/fetch_industry.py csrc --start 2007-12-01 --end 2022-12-31   # 证监会行业月度快照
   python scripts/fetch_industry.py cninfo                                    # 申万等行业变更记录（可断点续传）
   python scripts/build_relations.py --out /data/dataset/MASTER_relations --data_out data --qlib_dir <qlib_bin>
   python scripts/build_returns.py   --out /data/dataset/MASTER_relations --qlib_dir <qlib_bin>
   ```

各脚本的默认路径均为 `/data/dataset/...`，可以用 `--data_dir`、`--relation_dir`、`--qlib_dir` 等参数覆盖。

## 训练与评估

```bash
# MASTER 基准与 MARS（行业 + 相关图），一次训练同时记录官方准则与验证集选模两种协议
python main.py --universe csi300 --seeds 0 --tag base
python main.py --universe csi300 --seeds 0 --tag ind_corr --relations ind,corr
python main.py --universe csi300 --seeds 0 --tag multi    --relations ind,corr,fine

# 动态相关图（任意窗口 / 带符号边），固定 40 轮并保存每轮预测
python run_traj.py --universe csi300 --seed 0 --protocol full --relations ind,dcorr \
    --model_kwargs '{"corr_window": 60}' --tag ic_w60

# 两阶段 Student-t 不确定性（基于已训练的多关系模型）
python stage2_uncertainty.py --universe csi300 --seed 0 --stage1_tag multi --relations ind,corr,fine

# Qlib 回测：run_backtests.py 为官方设定（top-30，n_drop=30）；
# 换手约束回测用 mars.backtest.qlib_backtest(score, universe, start, end, n_drop=3)，见 scripts/final_s.py
python scripts/run_backtests.py --results results --out analysis_out/backtest.json
```

可用的关系类型（`--relations`）：

| 关系 | 含义 |
| --- | --- |
| `ind` | 证监会行业，按时点有效 |
| `corr` | 60 日 Top-10 相关图（预计算） |
| `dcorr` | 动态相关图，窗口与符号由 `--model_kwargs` 的 `corr_window`、`corr_mode`、`corr_edge` 指定（仅 `run_traj.py`） |
| `fine` / `sw1` / `sw3` | 申万二 / 一 / 三级行业 |
| `snap` | 用当前行业快照回填历史（对照） |
| `rand` | 随机图（对照） |

## 致谢与引用

MARS 基于 MASTER 的代码与数据。使用本仓库时，请引用 MASTER：

```latex
@inproceedings{li2024master,
  title={Master: Market-guided stock transformer for stock price forecasting},
  author={Li, Tong and Liu, Zhaoyang and Shen, Yanyan and Wang, Xue and Chen, Haokun and Huang, Sen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={162--170},
  year={2024}
}
```

许可证沿用原仓库的 MIT License（见 `LICENSE`）。
