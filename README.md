```markdown
# Predicting the Equity Premium with Supervised Learning and Frequency-Decomposed Variables

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/)

## Overview

The project bridges **signal processing** and **financial machine learning** by applying the Maximal Overlap Discrete Wavelet Transform (MODWT) to decompose 14 macroeconomic predictors into distinct frequency bands, capturing economic cycles from 2 months to over 10 years.

## Key Results

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

## 🚀 Research Contribution

### What Makes This Project Unique?

1. **Novel Methodology**: First comprehensive study combining MODWT wavelet decomposition with modern ensemble methods for equity premium prediction

2. **Frequency-Specific Insights**: Identifies that **short to medium-term periodicities (2-32 months)** contain the most predictive power

3. **Production-Ready Implementation**:
   - 104 models evaluated (98 univariate + 6 multivariate)
   - Parallel processing with ~75% runtime reduction
   - Comprehensive statistical testing framework
   - Bayesian hyperparameter optimization

## 🛠️ Technical Architecture

```
Data Pipeline
├── Goyal-Welch Dataset (1934-2024)
└── 14 Predictors × Monthly Frequency
    │
    ▼
MODWT Wavelet Decomposition
├── D1: 2-4 months (market noise)
├── D2: 4-8 months (seasonal effects)
├── D3: 8-16 months (short business cycles)
├── D4: 16-32 months (annual cycles)
├── D5: 32-64 months (medium business cycles)
├── D6: 64-128 months (long business cycles)
└── S6: >128 months (secular trends)
    │
    ▼
Machine Learning Models
├── Univariate OLS (98 models)
├── Multivariate OLS: unconstrained & constrained
├── Regularized: LASSO, Ridge, Elastic Net
├── Tree-based: Random Forest, LightGBM
└── Monthly re-estimation with expanding windows
```

## 📈 Key Findings

**1. Frequency Decomposition Enhances Prediction**
- 13 out of 14 predictors show improved forecasting accuracy
- Business cycle frequencies (8-32 months) consistently selected

**2. Machine Learning Superiority**
- LightGBM captures cross-frequency dependencies
- Ensemble averaging improves robustness
- Time-varying predictor relevance revealed

**3. Economic Significance**
- 8.29% annualized CER gain (fee investors would pay for access)
- Consistent outperformance across market regimes
- Results robust to transaction costs

## Repository Structure

```
equity-premium-wavelets/
├── MA-ERP.py                   # Main implementation
├── ThesisJonahSchmidt.pdf      # Complete thesis
├── Thesis-Proposal.pdf         # Research proposal
├── data/
│   └── PredictorData2023.xlsx  # Goyal-Welch dataset
├── results/
│   ├── comprehensive_report/    # Analysis outputs
│   └── performance_summary.csv  # Model comparison
└── notebooks/
    └── exploratory_analysis.ipynb
```

## Dataset

**Goyal-Welch (2024) Dataset** - Gold standard for equity premium research:
- **Period**: 1934-2024 (90 years monthly)
- **Target**: S&P 500 excess returns
- **Predictors**: 14 macroeconomic variables
  - Valuation ratios: D/P, D/Y, E/P, B/M
  - Interest rates: TBL, LTY, LTR
  - Spreads: TMS, DFY, DFR
  - Economic indicators: INFL, NTIS, RVOL

## Key References

- Goyal & Welch (2008): *A comprehensive look at equity premium prediction*
- Faria & Verona (2024): *Time-frequency forecast combinations*
- Campbell & Thompson (2008): *Predicting excess stock returns*

## Acknowledgments

Supervised by Prof. Dr. Emanuel Moench and Prof. Dr. Grigory Vilkov at Frankfurt School of Finance & Management.

## 📧 Contact

**Jonah Schmidt**  
Master of Science in Finance  
Frankfurt School of Finance & Management  
mail@jonah-schmidt.de

---
*Academic research project completed as Master's thesis requirement. Code designed for reproducibility and academic extension.*
```
