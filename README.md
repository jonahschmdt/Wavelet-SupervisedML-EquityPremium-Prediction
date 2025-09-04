```markdown
# Predicting the Equity Premium with Supervised Learning and Frequency-Decomposed Variables

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/)

## 🎯 Overview

The project bridges **signal processing** and **financial machine learning** by applying the Maximal Overlap Discrete Wavelet Transform (MODWT) to decompose 14 macroeconomic predictors into distinct frequency bands, capturing economic cycles from 2 months to over 10 years.

## 📊 Key Results

Our best-performing model (LightGBM with wavelet-decomposed features) achieves:

| Metric | Value | Significance |
|--------|-------|--------------|
| **Out-of-Sample R²** | 6.56% | *** (p < 0.01) |
| **CER Gain** | 8.29% p.a. | Economic significance |
| **Sharpe Ratio** | 0.90 | Risk-adjusted outperformance |
| **Evaluation Period** | 1975-2024 | 49 years out-of-sample |

### Performance Comparison

| Model | R² OOS (%) | CER Gain (%) | 
|-------|------------|--------------|
| **LightGBM** | **6.56*** | **8.29** |
| Elastic Net | 0.35* | 2.00 |
| LASSO | 0.13* | 1.70 |
| Ridge | -0.53 | 2.18 |
| Random Forest | -1.88 | 1.00 |
| OLS (baseline) | -1.45 | 2.31 |

*Statistical significance: *** p<0.01, ** p<0.05, * p<0.10*

## Research Contribution

### What Makes This Project Unique?

1. **Novel Methodology**: First comprehensive study combining MODWT wavelet decomposition with modern ensemble methods for equity premium prediction
   
2. **Frequency-Specific Insights**: Identifies that **short to medium-term periodicities (2-32 months)** contain the most predictive power for equity returns

3. **Production-Ready Implementation**: 
   - 104 models evaluated (98 univariate + 8 multivariate)
   - Parallel processing with ~75% runtime reduction
   - Comprehensive statistical testing framework
   - Bayesian hyperparameter optimization

## Technical Architecture

```
┌─────────────────────────────────────────────────┐
│                 Data Pipeline                   │
├─────────────────────────────────────────────────┤
│  Goyal-Welch Dataset (1934-2024)                │
│  14 Predictors × Monthly Frequency              │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│           MODWT Wavelet Decomposition           │
├─────────────────────────────────────────────────┤
│  • D1: 2-4 months (market noise)                │
│  • D2: 4-8 months (seasonal effects)            │
│  • D3: 8-16 months (short business cycles)      │
│  • D4: 16-32 months (annual cycles)             │
│  • D5: 32-64 months (medium business cycles)    │
│  • D6: 64-128 months (long business cycles)     │
│  • S6: >128 months (secular trends)             │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│            Machine Learning Models               │
├──────────────────────────────────────────────────┤
│  • Univariate OLS (112 models)                   │
│  • Multivariate OLS: unconstrained & constrained │
│  • Regularized: LASSO, Ridge, Elastic Net        │
│  • Tree-based: Random Forest, LightGBM           │
│  • Monthly re-estimation with expanding windows  │
└──────────────────────────────────────────────────┘
```

## 📈 Key Findings

### 1. Frequency Decomposition Enhances Prediction
- 13 out of 14 predictors show improved forecasting accuracy in frequency domain
- Business cycle frequencies (8-32 months) consistently selected by regularized models

### 2. Machine Learning Superiority
- LightGBM effectively captures **cross-frequency dependencies**
- Ensemble averaging across multiple seeds improves robustness
- Feature importance analysis reveals time-varying predictor relevance

### 3. Economic Significance
- **8.29% annualized CER gain** represents the fee investors would pay for model access
- Mean-variance portfolio achieves consistent outperformance across market regimes
- Results robust to transaction costs and portfolio constraints

## 🔧 Installation & Usage

### Prerequisites
```bash
python >= 3.8
numpy >= 1.20.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
lightgbm >= 3.0.0
pywavelets >= 1.1.0
optuna >= 2.10.0
statsmodels >= 0.12.0
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/equity-premium-wavelets.git
cd equity-premium-wavelets

# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python MA-ERP.py
```

### Core Functions
```python
# Wavelet decomposition
decomposed_data = modwt_decomposition(
    predictor_data,
    wavelet='haar',
    max_level=6
)

# Model training with univariate pre-screening
model_results = run_multivariate(
    data=df,
    model_type='lightgbm',
    use_bayesian=True,
    n_bayesian_trials=30
)

# Portfolio construction
portfolio_metrics = calculate_portfolio_performance(
    forecasts=model_results['forecasts'],
    returns=equity_premium,
    gamma=3  # Risk aversion
)
```


## Data

The project uses the **Goyal-Welch (2024) dataset**, the gold standard for equity premium prediction research:

- **Time Period**: 1934-2024 (90 years)
- **Frequency**: Monthly observations
- **Target**: S&P 500 excess returns
- **Predictors**: 14 macroeconomic variables including:
  - Valuation ratios (D/P, D/Y, E/P, B/M)
  - Interest rates (TBL, LTY, LTR)
  - Spreads (TMS, DFY, DFR)
  - Economic indicators (INFL, NTIS, RVOL)

## Academic Contribution

This research advances the empirical finance literature by:

1. **Extending univariate wavelet methods** (Faria & Verona, 2018-2024, Stein, 2024 and others) to multivariate settings
2. **Combining frequency decomposition with modern ML** for the first time in equity premium prediction
3. **Providing comprehensive empirical evidence** across 8 model architectures
4. **Establishing frequency-specific predictive content** in economic variables

## References

Key papers this work builds upon:

- Goyal, A., & Welch, I. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*
- Faria, G., & Verona, F. (2024). Time-frequency forecast combinations for the equity premium. *Journal of Financial Markets*
- Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample. *Review of Financial Studies*

## Acknowledgments

Special thanks to Prof. Dr. Emanuel Moench and Prof. Dr. Grigory Vilkov at Frankfurt School for their invaluable guidance and supervision.

## Contact

**Jonah Schmidt**  
Master of Science in Finance  
Frankfurt School of Finance & Management  
Email: mail@jonah-schmidt.de

---
*This project represents rigorous academic research conducted as part of my Master's thesis. The code is designed for reproducibility and extension by the research community.*
