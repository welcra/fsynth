import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CorporateGenes:
    ticker: str
    sector: int
    growth_potential: float
    margin_stability: float
    leverage_tolerance: float

class FundamentalGenerator:
    def __init__(self, market_data: pd.DataFrame, stock_metadata: pd.DataFrame, seed: int = 42):
        self.market = market_data
        self.stocks = stock_metadata
        self.rng = np.random.default_rng(seed)
        self.genes = self._assign_genes()
        
    def _assign_genes(self) -> Dict[str, CorporateGenes]:
        genes = {}
        for row in self.stocks.itertuples():
            if row.Sector == 0:
                growth = self.rng.normal(0.15, 0.05)
                margin = self.rng.normal(0.20, 0.10)
                lev = self.rng.normal(0.3, 0.1)
            elif row.Sector == 1:
                growth = self.rng.normal(0.03, 0.01)
                margin = self.rng.normal(0.10, 0.02)
                lev = self.rng.normal(0.6, 0.1)
            else:
                growth = self.rng.normal(0.05, 0.03)
                margin = self.rng.normal(0.15, 0.05)
                lev = self.rng.normal(0.4, 0.2)
                
            genes[row.Ticker] = CorporateGenes(
                ticker=row.Ticker,
                sector=row.Sector,
                growth_potential=growth,
                margin_stability=margin,
                leverage_tolerance=lev
            )
        return genes

    def generate_reports(self) -> pd.DataFrame:
        self.market['Quarter'] = self.market['Date'].dt.to_period('Q')
        quarter_ends = self.market.groupby('Quarter')['Date'].max().values
        quarter_regimes = self.market.groupby('Quarter')['Regime'].mean().values

        reports = []

        for ticker, gene in self.genes.items():
            revenue = self.rng.uniform(100, 1000)
            eps = revenue * 0.1 / self.rng.uniform(10, 50) 
            
            for q_idx, date in enumerate(quarter_ends):
                regime_severity = quarter_regimes[q_idx]
                
                macro_growth = 0.02 if regime_severity < 0.2 else -0.05
                macro_margin_impact = 0.0 if regime_severity < 0.2 else -0.05
                
                idio_perf = self.rng.normal(0, 0.05)
                
                growth_rate = gene.growth_potential + macro_growth + idio_perf
                revenue *= (1 + growth_rate)
                
                gross_margin = gene.margin_stability + macro_margin_impact + self.rng.normal(0, 0.01)
                gross_margin = np.clip(gross_margin, 0.05, 0.80)
                
                ebitda = revenue * gross_margin
                
                debt = (ebitda * 4) * (1 + regime_severity) 
                interest_expense = debt * (0.05 + (0.04 * regime_severity))
                
                net_income = ebitda - interest_expense
                shares_outstanding = 50
                eps = net_income / shares_outstanding
                
                reports.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Sector': gene.sector,
                    'Revenue': round(revenue, 2),
                    'EBITDA': round(ebitda, 2),
                    'NetIncome': round(net_income, 2),
                    'EPS': round(eps, 2),
                    'Debt': round(debt, 2),
                    'RegimeEnv': round(regime_severity, 2)
                })
                
        return pd.DataFrame(reports)