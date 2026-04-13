# Stock Elephant -- Part IV Empirical Data

@Vinodh\_Rag / Speculativa

Empirical data and scripts supporting **The Stock Elephant, Part IV: The Curse of the Reserve Currency**.

Published at: https://speculativa.substack.com

---

## Key Findings

### AmeriCo Granger Causality (1976-2025)

- **Globalization era (1990-2009):** NIIP deterioration causes manufacturing decline p=0.001***, causes savings depletion p=0.000***
- **Financialization era (2006-2025):** NIIP deterioration causes equity valuation expansion p=0.002***, causes rate suppression p=0.048**
- **Structural break:** Manufacturing elasticity collapsed 78% post-2016 -- the real adjustment mechanism is exhausted
- **Mediation shift:** Rate channel (early era) to real income channel (financialization era)

### Japan/PRC Surplus Side Granger (1976-2024)

**Japan 1976-1995:**
- H2J: FX undervaluation causes CA surplus p=0.0001***
- H3J: FX appreciation causes export decline p=0.0048***
- H4J: Export decline causes consumption reversal p=0.0142**

**PRC 1997-2024:**
- H1C REVERSED: consumption suppression upstream of surplus p=0.0397**
- H8C: WAP decline causes consumption share rising p=0.0037*** (demographic forcing)
- H4C NULL: managed rate severs FX transmission

### Trade Routing Analysis

- Transit/China import ratio: 94% (2017) to 326% (2025)
- Granger: Thailand p=0.000 lag=1m, Malaysia p=0.001 lag=2m (causal routing confirmed)
- Value-add ratios: Thailand 0.27x, Vietnam 0.54x, Malaysia 0.58x vs Mexico 1.36x (genuine)
- Customs wedge 2025: $113B annual. Cumulative post-inversion: $300B
- Income routing gap: $425B annual, $4.1T cumulative 2010-2024
- See also: https://github.com/speculativa/stock-elephant-routing

### Monte Carlo (kappa-SFC, 200,000 runs)

- Gate fires 100% of runs across all TFP levels -- structural, not cyclical
- Full break requires commodity shock: Hormuz 60%, post-ceasefire 97%, combined 100%+
- No-shock scenario: gate fires but system stabilizes -- buffers absorb without cascade
- Tail yield in full-break runs: 918-946 bps above base
- Falsification threshold: TFP above 13.3% eliminates structural divergence (not cascade)

---

## Repository Structure

```
data/
  granger/
    kappa_granger_results_v5.csv      -- AmeriCo Granger results all windows
    kappa_mediation_results_v5.csv    -- Mediation analysis results
    kappa_panel_data_v5.csv           -- Panel data 1976-2025
    kappa_japan_prc_granger_v2.csv    -- Japan/PRC surplus side Granger
  monte_carlo/
    mc_results_v1.csv                 -- Monte Carlo results 200K runs
    mc_fanchart.png                   -- Fan chart: break probability, tail yield
  routing/
    (see speculativa/stock-elephant-routing)

scripts/
  kappa_granger_colab_v5.py           -- AmeriCo Granger analysis
  kappa_japanprc_colab_v2.py          -- Japan/PRC Granger analysis
  kappa_mc_colab.py                   -- Monte Carlo engine
  china_routing_colab.py              -- Trade routing analysis
```

---

## Data Sources

- FRED (IIPUSNETIQ, DGS10, PSAVERT, MANEMP, NETFI, WALCL, FYGFDPUN)
- Shiller CAPE data: http://www.econ.yale.edu/~shiller/data.htm
- BIS Locational Banking Statistics Table A6 (Q4 2024)
- IMF Balance of Payments Statistics
- US Census Bureau USA Trade Online
- UN World Population Prospects 2024
- CFTC Traders in Financial Futures

---

## Citation

Raghunathan, V. (2026). "The Stock Elephant, Part IV: The Curse of the Reserve Currency."
Speculativa. https://speculativa.substack.com

---

*Data last updated: 2026-04-13*
