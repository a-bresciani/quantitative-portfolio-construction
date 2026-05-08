# Quantitative Portfolio Construction


---

## What this is

A multi-asset allocation framework built in Python across 15 sections, starting from covariance estimation and frontier construction, then extending into regime detection and a walk-forward backtest.


The central question: does conditioning the mean-variance allocation on inferred market regimes beat a static benchmark out-of-sample?

**It doesn't. That's documented and explained.**

---

## Libraries

```
numpy, pandas, scipy, scikit-learn, cvxpy, plotly, matplotlib,
seaborn, statsmodels, empyrical, arch, pandas-datareader,
networkx, openpyxl, pyarrow, tqdm, ipympl
```

Key modules:
- `sklearn.covariance.LedoitWolf` - shrinkage estimation
- `sklearn.cluster.KMeans` + `sklearn.metrics` - regime clustering, silhouette, Davies-Bouldin
- `cvxpy` (SCS solver) - QP frontier construction
- `scipy.cluster.hierarchy` + `scipy.spatial.distance` - HRP
- `scipy.ndimage.zoom` - bicubic interpolation for animated correlation surface
- `plotly.graph_objects` - all interactive charts

---

## Data

| File | Content |
|------|---------|
| `PriceClose dataset.csv` | 12 ETFs, daily closes, 2016-2026 |
| `VIX_dataset.csv` | Daily VIX |
| `US10YT_RR_dataset.csv` | US 10Y yield |
| `US2YT_RR_dataset.csv` | US 2Y yield |
| `REC_Da_dataset.csv` | NBER recession indicator (USRECDM) |

**Asset universe:**

| Ticker | Class |
|--------|-------|
| SPY | US Large-Cap Equity |
| IWM | US Small-Cap Equity |
| VEA | Developed ex-US Equity |
| VWO | Emerging Markets Equity |
| IEF.O | US Treasury 7-10Y |
| LQD | IG Corporate Bonds |
| HYG | HY Corporate Bonds |
| TIP | TIPS |
| VNQ | US REITs |
| PSP | Listed Private Equity |
| WOOD.O | Global Timber |
| IGF.O | Global Infrastructure |

---

## Structure

```
PriceClose_dataset_analysis_v2.ipynb   # full notebook, 124 cells
PriceClose dataset.csv
VIX_dataset.csv
US10YT_RR_dataset.csv
US2YT_RR_dataset.csv
REC_Da_dataset.csv
outputs/
    features_raw.parquet
    features_std.parquet
    regime_labels.parquet
    regime_centroids.parquet
    backtest_results.parquet
    performance_summary.parquet
    efficient_frontier.html
    stress_tests_comparison.html
    diversification_test.html
    risk_aversion_allocation.html
    hrp_3d_topology.html
    harmonic_dashboard_real_data.html
    *.csv  (weights, stress summary, regime labels)
```

---

## Sections 1-12: Core framework

**Sections 1-3: Data and setup**
- Monthly resampling, log returns converted to arithmetic
- Rf: time-weighted average of 1M T-bill rate 2016-2026 = 2.31%
- Reduced Rf = 0.50% used for tangency (at 2.31% the long-only optimizer concentrates 99%+ in SPY)

**Section 4: Covariance estimation**
- Ledoit-Wolf shrinkage: delta = 0.0557, condition number 1,079 to 137
- Positive definiteness: spectral check (all eigenvalues > 1e-8) + Sylvester's criterion (all 12 principal minors pass)

**Section 5: Monte Carlo**
- 25,000 portfolios, mixed Dirichlet (alpha = 1.0, 0.25, 0.1) to populate simplex boundaries
- Vectorized via `np.einsum`

**Section 6: Specific portfolios**
- GMV and Tangency (MC and analytical QP)
- CVaR 95%, Max Drawdown, realized Sharpe, turnover and TC drag
- Comparison: EW, GMV, Tangency

**Section 7: HRP**
- Angular distance matrix, single-linkage clustering, quasi-diagonalization, recursive bisection
- 3D interactive correlation surface

**Section 8: Multi-horizon correlation dynamics**
- Rolling 60-month correlation snapshots
- Regime clustering on flattened upper-triangular correlation matrices
- Horizons: 12, 36, 60 months

**Section 9: Efficient frontier**
- Analytical frontier via QP, curvature-adaptive density (more points near GMV)
- CAL built on analytical tangent Sharpe to guarantee tangency property

**Section 10: Full optimization (Sections A-E)**
- Unconstrained vs long-only frontier overlay with MC cloud
- Stress tests: vol +5pp, correlation shock (rho_new = min(rho+0.25, (rho+1)/2)), combined
- Asset removal: IEF.O (lowest avg correlation = 0.259), 12 vs 11 asset comparison
- Risk aversion: optimal CAL allocation for A = 1 to 15, U = E - 0.5 * A * sigma^2

**Section 11: Animated dashboard**
- 120 frames at 10-day step over 2,500 daily observations
- Top: rolling 3D correlation surface (252-day window, bicubic interpolation)
- Bottom: cumulative returns rebased to $100 with scan line

**Section 12: Cross-strategy performance**
- Realized backtest metrics: return, vol, Sharpe, Sortino, Max DD, CVaR
- Radar chart for strategy fingerprinting
- Strategies: GMV unconstrained, GMV long-only, Tangency unconstrained, Tangency long-only, HRP, EW

---

## Sections 13-15: Regime extension

**Section 13: Feature engineering**

9 rolling features, 60-day window, sampled at month-end:

| Feature | Captures |
|---------|---------|
| avg_correlation | mean pairwise correlation across 12 ETFs |
| dispersion_correlation | std of off-diagonal correlations |
| avg_realized_volatility | cross-sectional mean annualized vol |
| dispersion_volatility | cross-sectional std of vols |
| drawdown_depth | average distance from 60-day high |
| equity_bond_correlation | rolling SPY vs IEF.O correlation |
| momentum_60d | mean 60-day log return across ETFs |
| vix_level | end-of-period VIX |
| term_spread | US10Y minus US2Y |

Standardization: expanding-window z-score, min_periods=12. No look-ahead.

**Section 14: Regime clustering**
- K-means, k-means++, n_init=20, seed=42
- K=3: Calm / Inflation / Stress (deterministic semantic labeling rule)
- K selection: silhouette, Davies-Bouldin, inertia for K = 2 to 6
- K=2 has higher silhouette but collapses 2017 (zero-vol bull) and 2022 (inflation) into one bucket
- External validation: all 3 NBER recession months (COVID, Feb-Apr 2020) classified as Stress (3/3)
- Transition matrix: Calm 89% persistence, Stress 73% persistence

**Section 15: Walk-forward backtest**

Config:
```
WARMUP_MONTHS   = 24
TC_BPS          = 10
K_REGIMES       = 3
SHRINKAGE_ALPHA = 0.5   # James-Stein blend on mu
N_INIT          = 20
N_BOOTSTRAP     = 1000
```

7 strategies tested: dynamic tangency, static tangency, static GMV, equal weight (with and without 25% cap).

At each decision date t:
1. Re-fit K-means on features up to t (expanding window)
2. Predict regime for current month
3. Estimate moments using past returns in that regime only
4. James-Stein shrinkage on mu (alpha=0.5), Ledoit-Wolf on Sigma
5. Solve long-only tangency (cvxpy/SCS, optional 25% cap)
6. Realize return at t+1 net of TC

Bootstrap CI on Sharpe difference: 1,000 paired resamples, seed=42.
Sensitivity check: K=2, 3, 4 all yield the same conclusion (CI includes zero).

---

## Result

| Strategy | Sharpe | Max DD |
|---|---|---|
| Dynamic Tangency (cap 25%) | ~0.25 | ~-29% |
| Static Tangency (cap 25%) | ~0.25 | ~-24% |

Bootstrap 95% CI on Sharpe difference includes zero. p-value not significant.

Why it fails: ~10 Stress observations are not enough for stable covariance estimation in R^12. Regime-driven rebalancing runs at 2-3x the turnover of the static benchmark, costing ~3.5% of cumulative wealth at 10bps/trade over the backtest period. Consistent with Allen-Karjalainen (1999) and Bailey-Lopez de Prado (2014).

---

## How to run

```bash
pip install -r requirements.txt
jupyter notebook PriceClose_dataset_analysis_v2.ipynb
```

Run top-to-bottom. All outputs are generated in `outputs/`.

---

## Reproducibility

- Seed: 42 everywhere
- No look-ahead at any step
- All intermediate files saved to parquet for inspection
