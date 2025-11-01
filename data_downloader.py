#!/usr/bin/env python
# coding: utf-8
# %%
# --- Essential Libraries ---
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import concurrent.futures
import os

# --- Configuration ---
SECTORS_CSV_PATH = 'sp500_sectors.csv'
PRICES_CSV_PATH = 'sp500_prices.csv'
GLOBAL_START_DATE = '2000-01-01'
GLOBAL_END_DATE = '2025-10-31'

def fetch_and_save_sp500_data(start_date, end_date):
    """
    Fetches S&P 500 and essential strategy tickers' data and saves it to CSVs.
    """
    print("--- Starting Data Fetch and Caching Process ---")

    # Step 1: Scrape S&P 500 tickers from Wikipedia.
    print("Fetching S&P 500 ticker list from Wikipedia...")
    url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    table = pd.read_html(StringIO(response.text))[0]
    sp500_tickers = sorted([s.replace('.', '-') for s in table['Symbol'].tolist()])
    print(f"Found {len(sp500_tickers)} S&P 500 tickers on Wikipedia.")

    # Step 2: Add essential non-S&P 500 tickers required by the strategy.
    essential_tickers = ['SPY', '^IRX']
    print(f"Adding essential tickers: {', '.join(essential_tickers)}")
    all_tickers_to_download = sorted(list(set(sp500_tickers + essential_tickers)))

    # Step 3: Download historical price data for the combined list.
    print(f"Downloading historical price data for {len(all_tickers_to_download)} tickers...")
    data = yf.download(all_tickers_to_download, start=start_date, end=end_date, progress=True, auto_adjust=True)
    adj_close_prices = data['Close'].copy()

    # Step 4: Filter out tickers that failed to download.
    original_count = len(adj_close_prices.columns)
    adj_close_prices.dropna(axis=1, how='all', inplace=True)
    successful_tickers = adj_close_prices.columns.tolist()
    failed_count = original_count - len(successful_tickers)

    print(f"\nSuccessfully downloaded data for {len(successful_tickers)} tickers.")
    if failed_count > 0:
        print(f"Could not retrieve data for {failed_count} tickers; they will be excluded.")

    # Step 5: Fetch the sector for each valid ticker.
    print(f"\nFetching sector information for {len(successful_tickers)} tickers...")
    sectors_map = {}

    def _fetch_sector(ticker_symbol):
        """Helper to fetch sector for a single ticker."""
        try:
            info = yf.Ticker(ticker_symbol).info
            return ticker_symbol, info.get('sector', 'ETF/Index') # Assign a default category
        except Exception:
            return ticker_symbol, 'N/A'

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(_fetch_sector, successful_tickers)

    for ticker_symbol, sector in results:
        sectors_map[ticker_symbol] = sector
    print("Finished fetching sectors.")
    print("Assigning custom categories to essential tickers...")
    if 'SPY' in successful_tickers:
        sectors_map['SPY'] = 'ETF'
    if '^IRX' in successful_tickers:
        sectors_map['^IRX'] = 'Index'

    # Step 6: Save the data to CSV files.
    sectors_df = pd.DataFrame(list(sectors_map.items()), columns=['Ticker', 'Sector'])
    sectors_df.to_csv(SECTORS_CSV_PATH, index=False)
    print(f"\nTicker and sector data saved to '{SECTORS_CSV_PATH}'.")

    adj_close_prices.to_csv(PRICES_CSV_PATH)
    print(f"Historical price data saved to '{PRICES_CSV_PATH}'.")
    print("\n--- Data caching complete. You can now run the backtester script. ---")

if __name__ == "__main__":
    fetch_and_save_sp500_data(start_date=GLOBAL_START_DATE, end_date=GLOBAL_END_DATE)

