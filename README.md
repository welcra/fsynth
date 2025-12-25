# fsynth: High-Fidelity Synthetic Financial Data Generator

[![PyPI version](https://badge.fury.io/py/fsynth.svg)](https://badge.fury.io/py/fsynth)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

**fsynth** is a high-performance Python library for generating realistic, multi-asset financial time series and corresponding fundamental reports. Unlike simple geometric brownian motion (GBM) generators, `fsynth` models the complex statistical properties of real markets—including volatility clustering, fat tails, and regime-dependent correlations—using **Heston Stochastic Volatility** and **Merton Jump Diffusion** processes.

Designed for quantitative researchers, AI/ML engineers, and financial educators who need massive, clean, and statistically rigorous datasets for backtesting and model training.

---

## 🚀 Features

* **Stochastic Volatility:** Implements the Heston model to simulate time-varying volatility, capturing the "volatility smile" and clustering observed in real markets.
* **Regime Switching:** Simulates macro-economic states (Bull, Bear, Crisis) that dynamically alter correlation matrices and volatility baselines.
* **Jump Diffusion:** Incorporates Poisson-distributed price jumps to model market shocks (Merton model).
* **Linked Fundamentals:** Generates coherent 10-Q/10-K style fundamental data (Revenue, EBITDA, EPS, Debt) that correlates with the stock's price performance and sector genes.
* **High Performance:** Core simulation kernels are JIT-compiled using `numba` for C-level speeds, allowing for the generation of millions of rows in seconds.
* **Parquet Native:** Outputs optimized Parquet files ready for ingestion into Pandas, Polars, or PySpark.

---

## 📊 Why use fsynth? (The "Fat Tail" Problem)

Standard financial models (Geometric Brownian Motion) assume market returns are Normally Distributed. They fail to capture "Black Swan" events.

**fsynth is different.** It captures the "Fat Tails" (extreme crashes) observed in real markets.

![Fat Tail Analysis](images/spy5y_fsynth.png)

* **Real SPY Kurtosis:** ~8.04 (High risk of crash)
* **Standard Model:** ~0.00 (Assumes crashes are impossible)
* **fsynth Model:** ~5.81 (Successfully models crash risk)

--

## 📦 Installation

```bash
pip install fsynth
```

Or build from source:

```bash
git clone [https://github.com/welcra/fsynth.git](https://github.com/welcra/fsynth.git)
cd fsynth
pip install -e .
```

---

## ⚡ Quick Start

### 1. The Command Line Interface (CLI)

The easiest way to generate a dataset is using the bundled CLI tool. This command generates 500 stocks over 10 years and saves the data to the `data/` folder.

```bash
fsynth-gen --stocks 500 --years 10 --out data
```

**Output:**
* `data/market_index.parquet`: The macro-economic backbone (regimes, risk-free rates).
* `data/stock_prices.parquet`: OHLCV data for all 500 tickers (~1.2M rows).
* `data/fundamentals.parquet`: Quarterly financial reports for all tickers.

### 2. Python API

For integration into your own scripts or data pipelines:

```python
from fsynth import MarketConfig, MarketSimulator, FundamentalGenerator
import pandas as pd

# 1. Configure the Simulation
config = MarketConfig(
    T=5,                # Years
    dt=1/252,           # Daily time steps
    n_stocks=100,       # Number of tickers
    n_sectors=5,        # Distinct sectors with unique correlations
    seed=42             # Reproducibility
)

# 2. Run the Engine
sim = MarketSimulator(config)
print("Generating Market Backbone...")
market_df = sim.generate_market()

print("Generating Asset Paths...")
stock_dfs = sim.generate_stocks()

# 3. Aggregate Data
all_prices = pd.concat(stock_dfs.values(), ignore_index=True)
metadata = pd.DataFrame([
    {'Ticker': k, 'Sector': v['Sector'].iloc[0]} 
    for k, v in stock_dfs.items()
])

# 4. Generate Fundamentals
print("Generating 10-Q Reports...")
fund_gen = FundamentalGenerator(market_df, metadata, seed=config.seed)
fundamentals_df = fund_gen.generate_reports()

print(f"Generated {len(all_prices)} price rows and {len(fundamentals_df)} reports.")
```

---

## 🧮 Mathematical Methodology

`fsynth` moves beyond standard Random Walk theories to capture the nuanced risks of real financial markets.

### The Heston Model
We model the spot price $S_t$ and its variance $v_t$ using the following system of stochastic differential equations (SDEs):

$$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S$$

$$dv_t = \kappa(\theta - v_t)dt + \xi \sqrt{v_t} dW_t^v$$

Where:
* $\theta$ is the long-run average variance.
* $\kappa$ is the rate of mean reversion.
* $\xi$ is the volatility of volatility (vol-of-vol).
* $dW_t^S$ and $dW_t^v$ are Brownian motions with correlation $\rho$.

### Regime Switching & Jump Diffusion
To model market crashes and shocks (fat tails), we introduce a Poisson jump process $J$:

$$ \frac{dS_t}{S_t} = (\mu - \lambda k)dt + \sigma dW_t + dJ_t $$

A Hidden Markov Model (HMM) governs the transition between `Bull`, `Bear`, and `Crisis` regimes, automatically adjusting parameters $\mu$, $\sigma$, and jump intensity $\lambda$ in real-time.

---

## 📂 Data Structure

### Stock Prices (OHLCV)
| Date | Ticker | Open | High | Low | Close | Volume | Regime |
|------|--------|------|------|-----|-------|--------|--------|
| 2023-01-01 | STK_001 | 100.0 | 101.2 | 99.5 | 100.8 | 150240 | Bull |

### Fundamentals (Quarterly)
| Date | Ticker | Revenue | EBITDA | EPS | Debt | RegimeEnv |
|------|--------|---------|--------|-----|------|-----------|
| 2023-03-31 | STK_001 | 450.20 | 112.50 | 2.10 | 300.00 | 0.12 |

---

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss proposed changes or submit a Pull Request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Built by [Arnav Malhotra](https://github.com/welcra).**