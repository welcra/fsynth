import argparse
import pandas as pd
import os
import time
from .engine import MarketConfig, MarketSimulator
from .fundamentals import FundamentalGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic financial data.")
    parser.add_argument("--stocks", type=int, default=200, help="Number of stocks to simulate")
    parser.add_argument("--years", type=int, default=5, help="Years of data to generate")
    parser.add_argument("--out", type=str, default="output", help="Output directory")
    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    
    args = parser.parse_args()

    print(f"Initializing fsynth (Stocks: {args.stocks}, Years: {args.years})...")
    
    config = MarketConfig(
        T=args.years,
        dt=1/252,
        n_stocks=args.stocks,
        n_sectors=5,
        seed=args.seed
    )
    
    sim = MarketSimulator(config)
    
    print("  -> Generating Market Regime (Heston + Jump Diffusion)...")
    market_df = sim.generate_market()
    
    print("  -> Generating Individual Stock Paths...")
    stock_dfs = sim.generate_stocks()
    
    all_prices = []
    stock_metadata = []
    
    for ticker, df in stock_dfs.items():
        sector = df['Sector'].iloc[0]
        stock_metadata.append({'Ticker': ticker, 'Sector': sector})
        all_prices.append(df)
        
    price_master_df = pd.concat(all_prices, ignore_index=True)
    metadata_df = pd.DataFrame(stock_metadata)
    
    print("  -> Generatiing Fundamental Reports (10-K/10-Q style)...")
    fund_gen = FundamentalGenerator(market_df, metadata_df, seed=config.seed)
    fundamental_master_df = fund_gen.generate_reports()
    
    os.makedirs(args.out, exist_ok=True)
    
    print(f"  -> Saving Parquet files to {args.out}/...")
    market_df.to_parquet(f"{args.out}/market_index.parquet")
    price_master_df.to_parquet(f"{args.out}/stock_prices.parquet", compression='snappy')
    fundamental_master_df.to_parquet(f"{args.out}/fundamentals.parquet", compression='snappy')
    
    print(f"Done")

if __name__ == "__main__":
    main()