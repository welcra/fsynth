import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@njit(fastmath=True)
def heston_regime_switching_kernel(
    n_steps: int,
    dt: float,
    seed: int,
    mu0: float, kappa0: float, theta0: float, xi0: float,
    mu1: float, kappa1: float, theta1: float, xi1: float,
    p_01: float, p_10: float,
    lambda_j: float, mu_j: float, sigma_j: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    
    s = np.zeros(n_steps)
    v = np.zeros(n_steps)
    regimes = np.zeros(n_steps, dtype=np.int32)
    jumps = np.zeros(n_steps)
    
    s[0] = 100.0
    v[0] = theta0
    current_regime = 0
    
    for t in range(1, n_steps):
        rand_switch = np.random.random()
        if current_regime == 0:
            if rand_switch < p_01 * dt:
                current_regime = 1
        else:
            if rand_switch < p_10 * dt:
                current_regime = 0
        
        regimes[t] = current_regime
        
        mu = mu1 if current_regime == 1 else mu0
        kappa = kappa1 if current_regime == 1 else kappa0
        theta = theta1 if current_regime == 1 else theta0
        xi = xi1 if current_regime == 1 else xi0
        
        z1 = np.random.normal()
        z2 = np.random.normal()
        rho = -0.7 if current_regime == 1 else -0.3
        dw_s = z1
        dw_v = rho * z1 + np.sqrt(1 - rho**2) * z2
        
        curr_lambda = lambda_j * 3 if current_regime == 1 else lambda_j
        dn = np.random.poisson(curr_lambda * dt)
        jump_mag = 0.0
        if dn > 0:
            jump_mag = np.random.normal(mu_j, sigma_j) * dn
        jumps[t] = jump_mag
        
        dt_sqrt = np.sqrt(dt)
        v_prev = max(v[t-1], 1e-5)
        dv = kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev) * dw_v * dt_sqrt
        v[t] = np.abs(v_prev + dv)
        
        dr = (mu - 0.5 * v[t]) * dt + np.sqrt(v[t]) * dw_s * dt_sqrt + jump_mag
        s[t] = s[t-1] * np.exp(dr)
        
    return s, v, regimes, jumps, np.log(s[1:]/s[:-1])


@dataclass
class MarketConfig:
    T: int = 5
    dt: float = 1/252
    seed: int = 42
    n_stocks: int = 500
    n_sectors: int = 10

class MarketSimulator:
    def __init__(self, config: MarketConfig):
        self.cfg = config
        self.n_steps = int(config.T / config.dt)
        self.dates = pd.bdate_range(end=pd.Timestamp.today(), periods=self.n_steps)
        self.market_data = None
        self.market_returns = None
        
    def generate_market(self):
        s, v, regimes, jumps, rets = heston_regime_switching_kernel(
            self.n_steps, self.cfg.dt, self.cfg.seed,
            0.08, 2.0, 0.04, 0.3,
            -0.20, 4.0, 0.16, 0.6,
            0.1, 0.4,
            0.5, -0.05, 0.1
        )
        
        base_vol = 1e6
        volume = base_vol * (1 + (v/0.04)) * np.random.uniform(0.8, 1.2, self.n_steps)
        
        self.market_returns = rets
        
        noise_high = np.random.uniform(1.001, 1.01, self.n_steps)
        noise_low = np.random.uniform(0.99, 0.999, self.n_steps)
        
        df = pd.DataFrame({
            'Date': self.dates,
            'Close': s,
            'Volatility': np.sqrt(v),
            'Regime': regimes,
            'Volume': volume.astype(int)
        })
        df['Open'] = df['Close'].shift(1).fillna(df['Close'][0])
        df['High'] = df[['Open', 'Close']].max(axis=1) * noise_high
        df['Low'] = df[['Open', 'Close']].min(axis=1) * noise_low
        
        self.market_data = df
        return df

    def generate_stocks(self):
        if self.market_returns is None:
            raise ValueError("Run generate_market() first.")

        np.random.seed(self.cfg.seed)
        
        sectors = np.random.randint(0, self.cfg.n_sectors, self.cfg.n_stocks)
        
        sector_shocks = np.random.normal(0, 1, (self.n_steps-1, self.cfg.n_sectors))
        
        stock_dfs = {}
        
        idio_shocks = np.random.normal(0, 1, (self.n_steps-1, self.cfg.n_stocks))
        
        betas = np.random.normal(1.0, 0.3, self.cfg.n_stocks)
        alphas = np.random.normal(0.0, 0.0001, self.cfg.n_stocks)
        sigmas = np.random.uniform(0.15, 0.45, self.cfg.n_stocks)

        market_regimes = self.market_data['Regime'].values
        
        for i in range(self.cfg.n_stocks):
            sec_idx = sectors[i]
            
            r_stock = (alphas[i] + 
                       betas[i] * self.market_returns + 
                       0.5 * sector_shocks[:, sec_idx] * np.sqrt(self.cfg.dt) +
                       sigmas[i] * idio_shocks[:, i] * np.sqrt(self.cfg.dt))
            
            s0 = np.random.uniform(20, 200)
            price_path = np.zeros(self.n_steps)
            price_path[0] = s0
            price_path[1:] = s0 * np.exp(np.cumsum(r_stock))
            
            vol_stock = np.random.normal(1e5, 2e4, self.n_steps)
            
            df = pd.DataFrame({
                'Date': self.dates,
                'Close': price_path,
                'Volume': np.abs(vol_stock).astype(int),
                'Sector': sec_idx,
                'Ticker': f"STK_{i:04d}",
                'Regime': market_regimes
            })
            
            df['Open'] = df['Close'].shift(1).fillna(s0)
            df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, self.n_steps))
            df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, self.n_steps))
            
            stock_dfs[f"STK_{i:04d}"] = df
            
        return stock_dfs

if __name__ == "__main__":
    config = MarketConfig(T=2, dt=1/252, n_stocks=100)
    
    sim = MarketSimulator(config)
    
    market_df = sim.generate_market()
    print(market_df.head())
    print(f"Market Data Generated. Crises detected: {market_df['Regime'].sum()} days.")
    
    stocks = sim.generate_stocks()
    print(f"Generated {len(stocks)} stock tickers.")
    
    # market_df.to_parquet("market.parquet")