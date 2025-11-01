# Analysis of the Dynamic Core-Satellite Strategy

## 1. Introduction

This document provides a detailed analysis of the dynamic core-satellite investment strategy implemented in `backtester.py`. The strategy's primary objective is to outperform a passive SPY benchmark on a risk-adjusted basis over various market cycles.

The analysis is based on a series of 20 backtesting simulations, each starting on January 1st of a different year from 2006 to 2025, and all concluding on October 31, 2025. This approach allows for a robust examination of the strategy's behavior across different economic environments.

## 2. Detailed Methodology

The strategy is executed through a series of automated steps, from data processing to risk management.

### 2.1. Ticker Selection

The satellite portion of the portfolio is composed of stocks selected annually based on their historical risk-adjusted performance.

*   **Metric:** The annualized Sharpe Ratio is used as the selection metric. For the purpose of *selection*, the calculation assumes a risk-free rate of 0 to simplify the ranking of tickers based purely on their volatility-adjusted returns.
*   **Lookback Period:** The Sharpe Ratio for each stock is calculated using the prior **1 year** of historical price data (`YEARS_LOOKBACK_SELECTION = 1`).
*   **Universe & Diversification:** The selection universe is the S&P 500. To ensure sector diversification, the algorithm selects the top-performing stock(s) from each GICS sector.

### 2.2. Portfolio Optimization (Modern Portfolio Theory)

Once the satellite tickers are selected, Modern Portfolio Theory (MPT) is used to determine the optimal capital allocation.

*   **Objective Function:** The core of this step is an optimizer that seeks to find the portfolio weights that **maximize the portfolio's Sharpe Ratio**. This is achieved by minimizing the negative Sharpe Ratio using the `scipy.optimize.minimize` function.
*   **Optimizer:** The **SLSQP (Sequential Least Squares Programming)** algorithm is used, which is well-suited for constrained nonlinear optimization problems like this one.
*   **Lookback Period:** The MPT model is fed with a **36-month** rolling window of historical monthly returns to calculate the expected returns and covariance matrix (`LOOKBACK_PERIOD_MPT = 36`).
*   **Constraints:** Two critical constraints are applied to the optimization:
    1.  **Fully Invested:** The sum of all asset weights must equal 1.
    2.  **Diversification:** The weight of any single asset is capped at a maximum of **5%**. This is a crucial rule that prevents the portfolio from becoming overly concentrated in a few high-momentum stocks.

### 2.3. Risk Management & Rebalancing

The strategy incorporates two distinct mechanisms for managing risk and adjusting the portfolio over time.

*   **Drawdown Control:**
    *   **Trigger:** If the portfolio's total value falls **20%** below its all-time high (`DRAWDOWN_THRESHOLD = 0.20`), the risk management protocol is activated.
    *   **Action:** The algorithm identifies all assets that had a negative return in the most recent month. It then sells **10%** of the holdings of these underperforming assets (`DRAWDOWN_REDUCTION_FACTOR = 0.10`).
    *   **Reinvestment:** The proceeds from these sales are immediately reinvested into `SPY`, effectively reducing the portfolio's overall risk profile.

*   **Annual Rebalancing:**
    *   **Frequency:** The portfolio is rebalanced every year in January.
    *   **Smoothing Factor:** The rebalancing is not immediate. To reduce portfolio turnover and transaction costs, the strategy only moves **10%** of the way towards the newly calculated optimal weights (`ADJUSTMENT_FACTOR = 0.10`). This gradual adjustment process occurs over subsequent months.

## 3. Performance Analysis

The aggregated results of the 20 simulations reveal distinct patterns in the strategy's performance.

### 3.1. The Pre-2018 Environment (The "Golden Era")

In nearly every simulation initiated before 2018, the dynamic portfolio consistently outperformed the SPY benchmark on a risk-adjusted basis (Sharpe Ratio).

*   **Hypothesis:** This period, which included the recovery from the 2008 financial crisis, was characterized by broad market rallies where leadership was not confined to a few names. The strategy's methodology of picking top performers from each sector was highly effective in this environment. The built-in diversification (5% max weight) and risk management helped cushion the portfolio during volatile periods, contributing to a superior Sharpe Ratio.

### 3.2. The Post-2018 Environment (The "Concentration Challenge")

In simulations starting in 2018 or later, the strategy consistently underperformed the SPY benchmark.

*   **Hypothesis:** This market regime was heavily dominated by a small number of mega-cap technology stocks (e.g., FAANG). Because the SPY is a market-cap-weighted index, it benefited immensely from the meteoric rise of these few companies. The dynamic strategy's **5% diversification constraint**, while a strength in other periods, became a structural impediment to keeping pace. It was mathematically prevented from having the same level of exposure to the market's primary growth drivers.

### 3.3. The Most Recent Environment (The "Re-Validate")

In simulations starting in 2024, the strategy again outperformed the SPY.

*   **Hypothesis:** After the dramatic market shift centered around 2020, the market has returned to a stable upward trend similar to that before 2018. By selecting portfolio composition based on performance over the past year and rebalancing it annually by analyzing performance over the past three years, this strategy can more effectively select sustainable investment assets in a market environment that is returning to stability. However, the long-term performance and stability of portfolios built through this strategy need time to be verified.

### 3.4. Volatility Profile

Across most long-term simulations, the dynamic portfolio exhibited lower annualized volatility compared to the SPY benchmark. This suggests that the combination of MPT-based diversification and the drawdown control mechanism was effective at smoothing returns and managing risk, even if it sometimes came at the cost of capturing the full upside of a concentrated market.

## 4. Model Parameters & Assumptions

The strategy's behavior is highly dependent on a set of predefined parameters.

| Parameter                  | Value    | Impact                                                                                               |
| :------------------------- | :------- | :--------------------------------------------------------------------------------------------------- |
| `YEARS_LOOKBACK_SELECTION` | 1 year   | A shorter lookback makes the ticker selection more responsive to recent momentum.                    |
| `LOOKBACK_PERIOD_MPT`      | 36 months| A longer lookback for MPT provides more stable covariance estimates but may be slow to adapt.         |
| `DRAWDOWN_THRESHOLD`       | 20%      | Determines the sensitivity of the risk-management system. A lower value would trigger it more often. |
| `ADJUSTMENT_FACTOR`        | 10%      | Controls the speed of rebalancing, trading off between tracking the model and minimizing turnover.     |
| `TRANSACTION_COST_RATE`    | 0.05%    | Simulates brokerage fees, making the backtest more realistic.                                        |
| `CAPITAL_GAINS_TAX_RATE`   | 20%      | Simulates the impact of taxes on profitable trades, providing a more conservative return estimate.   |

## 5. Limitations & Biases

Every backtest has inherent limitations. It is crucial to acknowledge them for an objective assessment.

*   **Survivorship Bias:** As noted in the `README.md`, this is the most significant bias in this study. The stock universe is based on the current list of S&P 500 companies, which excludes firms that performed poorly and were delisted in the past. This inflates historical returns.
*   **Transaction Cost Simplification:** The model uses a fixed percentage for transaction costs. It does not account for more complex, real-world factors like **bid-ask spreads, slippage, or the market impact** of placing large orders.
*   **Static Parameters:** The model's parameters are fixed throughout the entire simulation period. In reality, an active manager might adjust these parameters based on changing market volatility or economic conditions (e.g., using a tighter drawdown threshold during a bear market).

## 6. Potential Future Work

This project provides a strong foundation that can be extended in several ways:

*   **Mitigating Survivorship Bias:** Incorporate historical point-in-time S&P 500 constituent lists to create a more historically accurate stock universe.
*   **Multi-Factor Ticker Selection:** Enhance the ticker selection model by moving beyond Sharpe Ratio to a multi-factor model that includes metrics for Value, Quality, and low Volatility.
*   **Alternative Optimization Models:** Implement and compare the performance of other allocation strategies, such as Risk Parity or the Black-Litterman model, against the current MPT approach.
*   **Dynamic Parameter Tuning:** Explore machine learning techniques to create an adaptive model where key parameters (like the drawdown threshold) adjust dynamically based on market volatility indicators (e.g., the VIX).
