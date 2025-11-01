# MPT-Based-Dynamic-Equity-Portfolio-Construction-And-Backtester
A quantitative portfolio framework in Python. Builds and backtests a dynamic equity strategy using MPT for allocation and a drawdown overlay for risk management. Simulates transaction costs and taxes for realistic performance analysis.

# Dynamic Core-Satellite Portfolio Strategy

This project develops and backtests a dynamic core-satellite investment strategy in Python. The model leverages Modern Portfolio Theory (MPT) for optimal asset allocation and incorporates a drawdown control mechanism for risk management.

To assess the strategy's robustness, its performance was simulated across **20 different market periods**.

## Backtesting Framework & Simulation Design

The project's core is a series of 20 distinct backtesting simulations designed to evaluate the strategy's performance across various market conditions.

*   **Multiple Simulations:** 20 separate backtests were conducted.
*   **Staggered Start Dates:** Each simulation begins on January 1st of a different year, starting in 2006 and concluding with the final simulation starting in 2025.
*   **Common End Date:** All 20 simulations run until the same end date: October 31, 2025.
*   **Objective:** This staggered-start approach allows for a comprehensive analysis of how the strategy performs when initiated during different market cycles.

## Strategy Methodology

Each of the 20 simulations employs the same core-satellite strategy.

*   **Ticker Selection (Satellite):** On an annual basis, the model selects a pool of top-performing stocks based on their historical Sharpe Ratio over the prior year.
*   **Portfolio Allocation (MPT):** The satellite portion is allocated using Modern Portfolio Theory, employing a `SciPy` optimizer to find the asset weights that maximize the Sharpe Ratio, subject to a 5% max allocation per asset.
*   **Risk Management (Drawdown Control):** The model activates a risk-reduction protocol if the portfolio's value falls 20% below its peak, selling underperforming assets and reinvesting proceeds into SPY.

## Aggregate Performance Summary

Across the 20 simulations, the strategy demonstrated a strong ability to generate superior risk-adjusted returns, particularly over longer time horizons.

*   **Risk-Adjusted Outperformance:** The dynamic portfolio achieved a higher Sharpe Ratio than the SPY benchmark in **12 out of 20** (60%) of the simulated periods.
*   **Absolute Return Outperformance:** In **10 of these 12** successful periods, the strategy not only managed risk better but also delivered a higher absolute annualized return (CAGR).

***Note:*** *Full, detailed metrics for all 20 simulation periods can be found in the `backtest_performance.log` log file.*

## Detailed Analysis

The summary above highlights the key performance outcomes. For a comprehensive breakdown of the strategy's methodology, a discussion of its performance drivers across different market regimes, and a full review of the model's parameters and limitations, please see the full report:

[**ANALYSIS.md**](ANALYSIS.md)

## Performance Charts

This repository contains the performance charts for all 20 simulations, comparing the dynamic portfolio against the SPY benchmark.

You can find all charts in the `/Charts` directory. Each file is named according to its simulation start date (e.g., `performance_chart_start_20060101.png`).

## A Note on Survivorship Bias

An important consideration in this backtest is the presence of **survivorship bias**. The stock universe is based on the current list of S&P 500 companies, which excludes firms that were delisted in the past due to poor performance. This inflates historical returns. A more rigorous backtest would use point-in-time constituent lists.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sigma-144/MPT-Based-Dynamic-Equity-Portfolio-Construction-And-Backtester.git
    cd MPT-Based-Dynamic-Equity-Portfolio-Construction-And-Backtester
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download and cache the data (run once):**
    ```bash
    python data_downloader.py
    ```

4.  **Run a backtest:**
    To run any of the 20 simulations, modify the `INVESTMENT_START_DATE` in `backtester.py` and execute the script.
    ```bash
    python backtester.py
    ```

## Technology Stack
*   **Python 3.x**
*   **Pandas & NumPy** for data manipulation.
*   **yfinance** for downloading market data.
*   **Matplotlib** for visualization.
*   **SciPy** for portfolio optimization.
