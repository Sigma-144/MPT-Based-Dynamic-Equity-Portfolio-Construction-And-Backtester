#!/usr/bin/env python
# coding: utf-8
# %%
# ==============================================================================
# --- Core-Satellite Backtesting and Analysis Script ---
# ==============================================================================
#
# Description:
# This script loads pre-cached S&P 500 data to run a backtest of a
# dynamic core-satellite investment strategy. The strategy is defined by:
#   1.  **Ticker Selection:** Selects top-performing stocks from each GICS sector
#       based on historical Sharpe Ratio.
#   2.  **Portfolio Allocation:** Uses a core-satellite approach with a small,
#       fixed allocation to a broad market ETF (SPY) and a risk-free asset (^IRX),
#       while the majority of capital is dynamically allocated to the selected
#       tickers using Modern Portfolio Theory (MPT) to maximize Sharpe Ratio.
#   3.  **Risk Management:** Implements a drawdown control mechanism that de-risks
#       the portfolio by selling underperforming assets and reinvesting the
#       proceeds into SPY when the portfolio value drops below a set threshold.
#
# How to Use:
# 1. Run the `data_downloader.py` script once to fetch and cache the necessary
#    S&P 500 price and sector data into local CSV files.
# 2. Adjust any of the settings in the "STRATEGY PARAMETERS" section below to
#    experiment with different configurations.
# 3. Run this script from your terminal: python backtester.py
#
# ==============================================================================


# %%
# --- Essential Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import os
import sys


# %%
# --- S T R A T E G Y   P A R A M E T E R S ---

# --- File Paths for Cached Data ---
SECTORS_CSV_PATH = 'sp500_sectors.csv'
PRICES_CSV_PATH = 'sp500_prices.csv'

# --- Ticker Selection Configuration ---
# The date the investment simulation begins. Tickers are selected based on performance *before* this date.
INVESTMENT_START_DATE = '2025-01-01'
# The number of years of historical data to use for calculating Sharpe Ratios for ticker selection.
YEARS_LOOKBACK_SELECTION = 1

# --- Portfolio & Model Configuration ---
# Core holdings and their fixed weights. SPY for market exposure, ^IRX for risk-free asset proxy.
CORE_HOLDINGS = {'SPY': 0.01, '^IRX': 0.01}
# Ticker for the risk-free asset (^IRX represents the 13-week Treasury Bill yield).
RISK_FREE_TICKER = '^IRX'
# Rolling lookback period (in months) for the Mean-Variance Optimization (MPT) model.
LOOKBACK_PERIOD_MPT = 36

# --- Backtest Date Configuration ---
# The end date for the entire backtesting simulation.
BACKTEST_END_DATE = '2025-10-31'

# --- Simulation Parameters ---
# The initial capital to be used for the backtest.
INITIAL_CAPITAL = 100000.00
# A flat rate applied to all transactions to simulate brokerage fees.
TRANSACTION_COST_RATE = 0.0005
# The tax rate applied to any capital gains realized from selling assets for a profit.
CAPITAL_GAINS_TAX_RATE = 0.20
# A factor that controls the speed of rebalancing towards the new optimal portfolio.
ADJUSTMENT_FACTOR = 0.10
# The drawdown percentage from the portfolio's peak value that will trigger the 
# Risk-reduction mechanism.
DRAWDOWN_THRESHOLD = 0.20
# The percentage of an underperforming asset's holding to sell during a drawdown event.
DRAWDOWN_REDUCTION_FACTOR = 0.10


# %%
# --- H E L P E R   F U N C T I O N S ---

def load_sp500_data_from_csv():
    """
    Loads the S&P 500 component data (prices and sectors) from local CSV files.
    This function is critical for separating data acquisition from analysis.
    """
    print("--- Loading Data from Cached CSV Files ---")
    if not os.path.exists(PRICES_CSV_PATH) or not os.path.exists(SECTORS_CSV_PATH):
        print("\nError: Data files not found.")
        print(f"Please run the `data_downloader.py` script first to generate '{PRICES_CSV_PATH}' and '{SECTORS_CSV_PATH}'.")
        sys.exit()  # Exit the script if data is missing

    # Load prices, ensuring the 'Date' column is parsed as dates and set as the index.
    price_data = pd.read_csv(PRICES_CSV_PATH, index_col='Date', parse_dates=True)
    # Load the ticker-to-sector mapping.
    sectors_df = pd.read_csv(SECTORS_CSV_PATH)

    # Reconstruct the sectors_map dictionary (sector -> list of tickers).
    sectors_map = defaultdict(list)
    for _, row in sectors_df.iterrows():
        sectors_map[row['Sector']].append(row['Ticker'])

    print("Successfully loaded data from local files.")
    return price_data, dict(sectors_map)

def calculate_sharpe_ratio(prices):
    """
    Calculates the annualized Sharpe Ratio for a pandas Series of prices.
    Note: For ticker selection, a risk-free rate of 0 is assumed for simplicity.
    """
    daily_returns = prices.pct_change().dropna()
    # Handle cases with no price volatility to avoid division by zero.
    if daily_returns.std() == 0:
        return 0.0
    # Annualize by multiplying by the square root of the number of trading days in a year.
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

def select_top_tickers_by_sharpe(price_data, sectors_map, analysis_end_date_str, years_lookback):
    """
    Analyzes pre-loaded price data to select the top tickers from each sector
    based on their Sharpe Ratio over a specified lookback period.
    """
    print(f"\n--- Starting Ticker Selection for Date: {analysis_end_date_str} ---")
    end_date = pd.to_datetime(analysis_end_date_str)
    start_date = end_date - pd.DateOffset(years=years_lookback)

    # Filter the master price data for the specified historical analysis period.
    period_data = price_data.loc[start_date:end_date]
    final_selection = []

    print(f"Analyzing performance over a {years_lookback}-year lookback period...")
    for sector, tickers in sectors_map.items():
        sector_sharpe_metrics = []
        for ticker in tickers:
            if ticker in period_data.columns:
                ticker_prices = period_data[ticker].dropna()
                # Ensure there's enough historical data (at least 80%) for a reliable calculation.
                if len(ticker_prices) >= 252 * years_lookback * 0.80:
                    sharpe = calculate_sharpe_ratio(ticker_prices)
                    sector_sharpe_metrics.append({'ticker': ticker, 'sharpe': sharpe})

        # Sort tickers within the sector by Sharpe ratio in descending order and take the top ones.
        top_performers = sorted(sector_sharpe_metrics, key=lambda x: x['sharpe'], reverse=True)[:12]
        selected_symbols = [item['ticker'] for item in top_performers]
        if selected_symbols:
            print(f"Top performers in {sector}: {', '.join(selected_symbols)}")
        final_selection.extend(selected_symbols)

    print("\nTicker selection process complete.")
    # Return a unique list of selected tickers.
    return sorted(list(set(final_selection)))

def get_optimal_portfolio(returns, last_known_risk_free_rate):
    """
    Calculates portfolio weights that maximize the Sharpe Ratio using MPT.
    This is the core of the dynamic allocation logic.
    """
    num_assets = len(returns.columns)
    # Annualize mean returns and the covariance matrix for the optimization.
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        """The objective function to be minimized (hence, the negative Sharpe Ratio)."""
        p_return = np.sum(mean_returns * weights)
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_return - risk_free_rate) / p_std if p_std != 0 else -np.inf

    # Constraint: The sum of all weights must be 1 (fully invested).
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Constraint: Each individual asset weight must be between 0% and 5% to ensure diversification.
    bounds = tuple((0.00, 0.05) for _ in range(num_assets))
    # Initial guess for the optimizer (equal weighting).
    initial_weights = np.array([1./num_assets] * num_assets)

    # Use the SLSQP (Sequential Least Squares Programming) optimizer.
    result = minimize(neg_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, last_known_risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def calculate_metrics(returns, risk_free_series):
    """Calculates and formats key performance metrics for a given returns series."""
    if returns.empty: return {"Error": "No data available."}

    cum_ret = (1 + returns).prod() - 1                                # Total return over the period.
    ann_ret = ((1 + returns).prod())**(12/len(returns)) - 1            # Geometric average annual return.
    ann_vol = returns.std() * np.sqrt(12)                             # Annualized standard deviation of returns.
    rf_mean = risk_free_series.reindex(returns.index).mean() * 12     # Average risk-free rate.
    sharpe = (ann_ret - rf_mean) / ann_vol if ann_vol != 0 else 0      # Risk-adjusted return.

    return {"Cumulative Return": f"{cum_ret:.2%}", "Annualized Return": f"{ann_ret:.2%}",
            "Annualized Volatility": f"{ann_vol:.2%}", "Sharpe Ratio": f"{sharpe:.4f}"}


# %%
# --- M A I N   E X E C U T I O N   F L O W ---
if __name__ == "__main__":
    # Step 1: Load Data and Select Initial Tickers
    master_price_df, sp500_sectors_map = load_sp500_data_from_csv()

    # Define categories to exclude from the dynamic ticker selection process.
    excluded_categories = ['ETF', 'Index']

    # Create a new, filtered dictionary that contains the actual stock sectors.
    stock_sectors_map = {
        sector: tickers
        for sector, tickers in sp500_sectors_map.items()
        if sector not in excluded_categories
    }

    top_tickers = select_top_tickers_by_sharpe(
        master_price_df,
        stock_sectors_map,
        INVESTMENT_START_DATE,
        YEARS_LOOKBACK_SELECTION
    )

    # Step 2: Finalize Asset Lists for the Backtest
    dynamic_tickers = list(set(top_tickers))
    # The MPT model optimizes the dynamic tickers plus the market ETF (SPY).
    optimization_tickers = sorted(list(set(dynamic_tickers + ['SPY'])))
    # The complete list of assets needed for the simulation includes the core holdings.
    all_tickers_for_backtest = sorted(list(set(dynamic_tickers + list(CORE_HOLDINGS.keys()))))

    # Step 3: Prepare DataFrames for the Specific Simulation Period
    investment_start_dt = pd.to_datetime(INVESTMENT_START_DATE)
    # Ensure data starts early enough to cover the initial MPT lookback period.
    data_start_date = (investment_start_dt - pd.DateOffset(years=2, months=1)).strftime('%Y-%m-%d')

    # Filter the master price dataframe for the assets and dates required for this specific backtest run.
    asset_prices = master_price_df.loc[data_start_date:BACKTEST_END_DATE, all_tickers_for_backtest].dropna(axis=0, how='all')
    monthly_prices = asset_prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()

    # Prepare the risk-free rate series, converting the annualized T-bill yield to a monthly rate.
    risk_free_rate = asset_prices[RISK_FREE_TICKER].ffill() / 100
    monthly_risk_free_rate = (1 + risk_free_rate)**(1/12) - 1
    monthly_risk_free_rate = monthly_risk_free_rate.reindex(monthly_returns.index).ffill()

    # Step 4: Run the Backtesting Engine
    print("\nStarting backtest simulation...")

    # --- Initialize State Trackers ---
    portfolio = {'cash': INITIAL_CAPITAL, 'total_value': INITIAL_CAPITAL, 'risk_free_value': 0.0}
    tradeable_tickers = [t for t in all_tickers_for_backtest if t != RISK_FREE_TICKER]
    portfolio['shares'] = {ticker: 0 for ticker in tradeable_tickers}
    portfolio['cost_basis'] = {ticker: 0 for ticker in tradeable_tickers}
    portfolio_history = pd.DataFrame(index=monthly_prices.index)
    is_invested = False
    peak_value = INITIAL_CAPITAL

    # --- Main Backtesting Loop ---
    for date in monthly_prices.index:
        if date < investment_start_dt:
            continue

        if is_invested:
            portfolio['risk_free_value'] *= (1 + monthly_risk_free_rate.loc[date])

        # --- Initial Investment ---
        if not is_invested:
            print(f"Performing initial investment on {date.date()}...")
            # Allocate to risk-free core holding.
            value_to_buy_rf = INITIAL_CAPITAL * CORE_HOLDINGS[RISK_FREE_TICKER]
            portfolio['cash'] -= value_to_buy_rf
            portfolio['risk_free_value'] += value_to_buy_rf

            # Optimize and allocate the dynamic satellite portion.
            dynamic_capital = INITIAL_CAPITAL - value_to_buy_rf
            lookback_start = date - pd.DateOffset(months=LOOKBACK_PERIOD_MPT)
            lookback_end = date - pd.DateOffset(days=1)
            lookback_data = monthly_returns.loc[lookback_start:lookback_end][optimization_tickers]
            last_risk_free = monthly_risk_free_rate.loc[:lookback_end].iloc[-1] * 12
            target_weights = get_optimal_portfolio(lookback_data, last_risk_free)

            # Enforce minimum weight for SPY.
            spy_index = optimization_tickers.index('SPY')
            min_spy_weight_dynamic = (INITIAL_CAPITAL * CORE_HOLDINGS['SPY']) / dynamic_capital
            if target_weights[spy_index] < min_spy_weight_dynamic:
                target_weights[spy_index] = min_spy_weight_dynamic
                other_weights_sum = np.sum(target_weights) - min_spy_weight_dynamic
                scale_factor = (1 - min_spy_weight_dynamic) / other_weights_sum
                for i in range(len(target_weights)):
                    if i != spy_index: target_weights[i] *= scale_factor

            target_weights /= np.sum(target_weights)

            # Execute initial buys.
            for i, ticker in enumerate(optimization_tickers):
                value_to_buy = dynamic_capital * target_weights[i]
                if value_to_buy > 0:
                    price = monthly_prices.loc[date, ticker]
                    cost = value_to_buy * TRANSACTION_COST_RATE
                    if portfolio['cash'] >= (value_to_buy + cost):
                        portfolio['cash'] -= (value_to_buy + cost)
                        portfolio['shares'][ticker] = value_to_buy / price
                        portfolio['cost_basis'][ticker] = price
            is_invested = True

        # --- Monthly Drawdown Risk Management ---
        asset_value = sum(portfolio['shares'][t] * monthly_prices.loc[date, t] for t in tradeable_tickers)
        current_total_value = portfolio['cash'] + portfolio['risk_free_value'] + asset_value
        peak_value = max(peak_value, current_total_value)
        drawdown = (peak_value - current_total_value) / peak_value

        if is_invested and drawdown > DRAWDOWN_THRESHOLD:
            print(f"--- Drawdown Alert on {date.date()}: {drawdown:.2%}. Reducing risk. ---")
            proceeds_for_reinvestment = 0
            underperformers = [t for t in tradeable_tickers if monthly_returns.loc[date, t] < 0]

            # Sell a portion of underperforming assets.
            for ticker in underperformers:
                value_to_sell = portfolio['shares'][ticker] * monthly_prices.loc[date, ticker] * DRAWDOWN_REDUCTION_FACTOR
                if value_to_sell > 0:
                    shares_to_sell = value_to_sell / monthly_prices.loc[date, ticker]
                    proceeds = shares_to_sell * monthly_prices.loc[date, ticker]
                    gain = (monthly_prices.loc[date, ticker] - portfolio['cost_basis'][ticker]) * shares_to_sell
                    tax = max(0, gain * CAPITAL_GAINS_TAX_RATE)
                    cost = proceeds * TRANSACTION_COST_RATE
                    proceeds_for_reinvestment += proceeds - cost - tax
                    portfolio['shares'][ticker] -= shares_to_sell

            # Reinvest proceeds into SPY.
            if proceeds_for_reinvestment > 0:
                spy_price = monthly_prices.loc[date, 'SPY']
                value_of_spy_to_buy = proceeds_for_reinvestment * (1 - TRANSACTION_COST_RATE)
                if value_of_spy_to_buy > 0:
                    shares_of_spy_to_buy = value_of_spy_to_buy / spy_price
                    old_spy_value = portfolio['shares']['SPY'] * portfolio['cost_basis']['SPY']
                    total_spy_shares = portfolio['shares']['SPY'] + shares_of_spy_to_buy
                    if total_spy_shares > 0:
                         portfolio['cost_basis']['SPY'] = (old_spy_value + value_of_spy_to_buy) / total_spy_shares
                    portfolio['shares']['SPY'] += shares_of_spy_to_buy

        # --- Regular Annual Rebalancing (occurs in January) ---
        if is_invested and date.month == 1:
            asset_value = sum(portfolio['shares'][t] * monthly_prices.loc[date, t] for t in tradeable_tickers)
            total_value = portfolio['cash'] + portfolio['risk_free_value'] + asset_value
            dynamic_value = total_value - portfolio['risk_free_value'] - portfolio['cash']
            current_dynamic_weights = np.array([(portfolio['shares'][t] * monthly_prices.loc[date, t]) / dynamic_value if dynamic_value > 0 else 0 for t in optimization_tickers])

            rebalance_lookback_end = date - pd.DateOffset(days=1)
            rebalance_lookback_start = date - pd.DateOffset(months=LOOKBACK_PERIOD_MPT)
            lookback_data = monthly_returns.loc[rebalance_lookback_start:rebalance_lookback_end][optimization_tickers]
            last_risk_free = monthly_risk_free_rate.loc[:rebalance_lookback_end].iloc[-1] * 12
            optimal_weights = get_optimal_portfolio(lookback_data, last_risk_free)

            unadjusted_target_weights = (1 - ADJUSTMENT_FACTOR) * current_dynamic_weights + ADJUSTMENT_FACTOR * optimal_weights
            target_weights = unadjusted_target_weights / np.sum(unadjusted_target_weights)

            target_holdings = {t: dynamic_value * target_weights[i] for i, t in enumerate(optimization_tickers)}
            current_holdings = {t: portfolio['shares'][t] * monthly_prices.loc[date, t] for t in optimization_tickers}
            trades = {t: target_holdings[t] - current_holdings[t] for t in optimization_tickers}

            # Process sells first to generate cash.
            for ticker, trade_value in trades.items():
                if trade_value < 0:
                    shares_to_sell = min(-trade_value / monthly_prices.loc[date, ticker], portfolio['shares'][ticker])
                    proceeds = shares_to_sell * monthly_prices.loc[date, ticker]
                    gain = (monthly_prices.loc[date, ticker] - portfolio['cost_basis'][ticker]) * shares_to_sell
                    tax = max(0, gain * CAPITAL_GAINS_TAX_RATE)
                    cost = proceeds * TRANSACTION_COST_RATE
                    portfolio['cash'] += proceeds - cost - tax
                    portfolio['shares'][ticker] -= shares_to_sell

            # Process buys with the available cash.
            for ticker, trade_value in trades.items():
                if trade_value > 0:
                    cost = trade_value * (1 + TRANSACTION_COST_RATE)
                    if portfolio['cash'] >= cost:
                        portfolio['cash'] -= cost
                        shares_to_buy = trade_value / monthly_prices.loc[date, ticker]
                        old_value = portfolio['shares'][ticker] * portfolio['cost_basis'][ticker]
                        total_shares = portfolio['shares'][ticker] + shares_to_buy
                        if total_shares > 0:
                            portfolio['cost_basis'][ticker] = (old_value + trade_value) / total_shares
                        portfolio['shares'][ticker] += shares_to_buy

        # --- Monthly Portfolio Value Update ---
        asset_value = sum(portfolio['shares'][t] * monthly_prices.loc[date, t] for t in tradeable_tickers)
        portfolio['total_value'] = portfolio['cash'] + portfolio['risk_free_value'] + asset_value
        portfolio_history.loc[date, 'value'] = portfolio['total_value']
        peak_value = max(peak_value, portfolio['total_value'])

    portfolio_history = portfolio_history.loc[investment_start_dt:].dropna()
    print("Backtest complete.")

# Step 5: Performance Analysis and Reporting
    strategy_returns = portfolio_history['value'].pct_change().dropna()
    benchmark_returns = monthly_returns['SPY'].reindex(strategy_returns.index)

    # --- Calculate metrics for both console display and logging ---
    strategy_metrics = calculate_metrics(strategy_returns, monthly_risk_free_rate)
    benchmark_metrics = calculate_metrics(benchmark_returns, monthly_risk_free_rate)

    print("\n--- Dynamic Portfolio (Core-Satellite with Drawdown Control) ---")
    print(strategy_metrics)

    print("\n--- SPY Benchmark (Buy & Hold) ---")
    print(benchmark_metrics)

    # --- File Paths for Outputs ---
    LOG_FILE_PATH = 'backtest_performance.log'
    CHART_OUTPUT_DIR = 'charts'

    # --- Create Output Directory if it Doesn't Exist ---
    if not os.path.exists(CHART_OUTPUT_DIR):
        os.makedirs(CHART_OUTPUT_DIR)
        print(f"\nCreated directory: '{CHART_OUTPUT_DIR}'")

    # --- Prepare Log Entry as a Formatted String ---
    log_entry = (
        f"==============================================================================\n"
        f"--- SIMULATION PERIOD: {investment_start_dt.strftime('%Y-%m-%d')} to {pd.to_datetime(BACKTEST_END_DATE).strftime('%Y-%m-%d')} ---\n"
        f"==============================================================================\n\n"
        f"--- Dynamic Portfolio (Core-Satellite with Drawdown Control) ---\n"
        f"{str(strategy_metrics)}\n\n"
        f"--- SPY Benchmark (Buy & Hold) ---\n"
        f"{str(benchmark_metrics)}\n\n"
    )

    # --- Write the Formatted String to the Log File ---
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(log_entry)

    print(f"--- Simulation results logged to '{LOG_FILE_PATH}' ---")

    # Step 6: Visualization and Chart Saving
    strategy_cumulative_returns = (1 + strategy_returns).cumprod()
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    (strategy_cumulative_returns * 100).plot(ax=ax, label='Dynamic Portfolio', lw=2)
    (benchmark_cumulative_returns * 100).plot(ax=ax, label='SPY Benchmark', lw=2, linestyle='--')

    chart_title = f'Portfolio Performance vs. Benchmark ({investment_start_dt.year}-{pd.to_datetime(BACKTEST_END_DATE).year})'
    ax.set_title(chart_title, fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (Initial $100)')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True)

    # --- Save the Figure to the Output Directory ---
    chart_filename = f"performance_chart_start_{investment_start_dt.strftime('%Y%m%d')}.png"
    chart_filepath = os.path.join(CHART_OUTPUT_DIR, chart_filename)
    plt.savefig(chart_filepath)
    plt.close(fig) # Close the figure to free up memory instead of displaying it.

    print(f"--- Performance chart saved to '{chart_filepath}' ---")

