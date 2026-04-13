# kappa_japanprc_colab_v2.py
# Japan/PRC Surplus Analysis -- Redesigned with proper mediation VAR
# Tests the rebalancing trap: consumption rise -> FX appreciation ->
#   export income fall -> consumption reversal
# Three-variable mediation + VAR IRF
# Vinodh Raghunathan / Speculativa / April 2026

# ── CELL 1: INSTALL ──────────────────────────────────────────────────────────
# !pip install fredapi statsmodels pandas numpy matplotlib seaborn -q

# ── CELL 2: IMPORTS ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os, io, json, urllib.request
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

from fredapi import Fred
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

# ── CELL 3: CONFIG ───────────────────────────────────────────────────────────

FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"
SAVE_DIR     = "/content/drive/MyDrive/StockElephant/"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Save directory: {SAVE_DIR}")

# ── CELL 4: HARDCODED ANCHOR DATA ────────────────────────────────────────────

# Japan CA surplus % GDP 1976-1995 (IMF BOP Historical)
JAPAN_CA = {
    1976:0.7, 1977:1.6, 1978:1.7, 1979:-0.9, 1980:-1.0,
    1981:0.4, 1982:0.6, 1983:1.8, 1984:2.8,  1985:3.7,
    1986:4.4, 1987:3.6, 1988:2.7, 1989:2.0,  1990:1.4,
    1991:2.0, 1992:3.0, 1993:3.1, 1994:2.8,  1995:2.1,
}

# Japan household consumption % GDP 1976-1995 (World Bank WDI)
JAPAN_CONS = {
    1976:58.2,1977:57.9,1978:57.5,1979:57.1,1980:56.8,
    1981:56.2,1982:55.8,1983:55.4,1984:54.9,1985:54.3,
    1986:53.8,1987:53.6,1988:53.2,1989:52.8,1990:52.5,
    1991:52.0,1992:51.8,1993:52.1,1994:52.5,1995:52.8,
}

# Japan gross savings % GDP 1976-1995 (World Bank WDI)
JAPAN_SAV = {
    1976:32.1,1977:31.8,1978:31.5,1979:31.2,1980:30.8,
    1981:30.5,1982:30.1,1983:29.8,1984:29.5,1985:29.1,
    1986:32.5,1987:33.1,1988:33.8,1989:34.2,1990:34.5,
    1991:34.1,1992:33.8,1993:33.2,1994:32.5,1995:31.8,
}

# PRC household consumption % GDP 1995-2024 (World Bank WDI)
PRC_CONS = {
    1995:44.9,1996:43.8,1997:42.5,1998:41.2,1999:40.1,
    2000:39.2,2001:38.5,2002:37.8,2003:36.5,2004:35.2,
    2005:34.1,2006:35.2,2007:36.1,2008:35.8,2009:35.2,
    2010:35.6,2011:36.2,2012:36.8,2013:37.1,2014:37.5,
    2015:38.0,2016:38.5,2017:38.8,2018:38.9,2019:39.1,
    2020:38.5,2021:38.9,2022:39.2,2023:39.5,2024:39.8,
}

# PRC gross savings % GDP 1995-2024 (World Bank WDI)
PRC_SAV = {
    1995:41.5,1996:42.8,1997:43.5,1998:44.2,1999:44.8,
    2000:37.5,2001:38.5,2002:40.2,2003:42.5,2004:44.8,
    2005:46.2,2006:48.5,2007:50.2,2008:51.5,2009:52.8,
    2010:51.2,2011:49.8,2012:48.5,2013:47.2,2014:46.5,
    2015:45.8,2016:45.2,2017:44.8,2018:44.5,2019:44.2,
    2020:44.8,2021:45.2,2022:45.8,2023:46.1,2024:46.5,
}

# ── DEMOGRAPHIC DATA (UN World Population Prospects 2024) ───────────────────

# China working age population 15-64 (millions) 1995-2024
PRC_WAP = {
    1995:859, 1996:867, 1997:875, 1998:883, 1999:891,
    2000:899, 2001:907, 2002:916, 2003:925, 2004:934,
    2005:942, 2006:950, 2007:958, 2008:965, 2009:971,
    2010:975, 2011:977, 2012:977, 2013:975, 2014:971,
    2015:966, 2016:960, 2017:954, 2018:948, 2019:942,
    2020:935, 2021:928, 2022:920, 2023:911, 2024:902,
}

# China old-age dependency ratio (population 65+ / population 15-64 * 100)
PRC_DEP = {
    1995:10.1,1996:10.3,1997:10.5,1998:10.7,1999:10.9,
    2000:11.1,2001:11.3,2002:11.5,2003:11.7,2004:11.9,
    2005:12.1,2006:12.3,2007:12.6,2008:12.9,2009:13.2,
    2010:13.5,2011:13.8,2012:14.2,2013:14.7,2014:15.2,
    2015:15.8,2016:16.4,2017:17.0,2018:17.7,2019:18.4,
    2020:19.2,2021:20.0,2022:20.9,2023:21.8,2024:22.8,
}

# Japan working age population 15-64 (millions) 1976-1995
JPN_WAP = {
    1976:72, 1977:73, 1978:74, 1979:75, 1980:76,
    1981:77, 1982:78, 1983:79, 1984:80, 1985:81,
    1986:82, 1987:83, 1988:84, 1989:85, 1990:86,
    1991:87, 1992:87, 1993:87, 1994:87, 1995:87,
}

# Japan old-age dependency ratio 1976-1995
JPN_DEP = {
    1976:12.4,1977:12.8,1978:13.2,1979:13.6,1980:14.0,
    1981:14.5,1982:14.9,1983:15.4,1984:15.8,1985:16.3,
    1986:16.9,1987:17.4,1988:17.9,1989:18.4,1990:19.0,
    1991:19.6,1992:20.3,1993:21.0,1994:21.7,1995:22.4,
}

def make_annual_series(d, name):
    s = pd.Series(
        {pd.Timestamp(f"{yr}-01-01"): val for yr, val in d.items()}
    )
    s.index = s.index.tz_localize(None)
    s.name = name
    return s

# ── CELL 5: DATA PULL ────────────────────────────────────────────────────────

def pull_all_data(api_key):
    fred = Fred(api_key=api_key)
    data = {}

    series_map = {
        # Japan
        "jpn_fx":      ("EXJPUS",            "1976-01-01"),  # JPY/USD
        "jpn_exports": ("XTNTVA01JPM667S",   "1976-01-01"),  # Net trade goods
        "jpn_wage":    ("LCEAMN01JPM661S",   "1976-01-01"),  # Mfg hourly earnings
        "jpn_wage2":   ("LCEAPR03JPM661S",   "1976-01-01"),  # Private sector earnings
        "jpn_cpi":     ("JPNCPIALLMINMEI",   "1976-01-01"),  # CPI
        "jpn_yield":   ("IRLTLT01JPM156N",   "1976-01-01"),  # 10yr JGB
        "jpn_ca_gdp":  ("JPNBCAGDPBP6PT",   "1990-01-01"),  # CA % GDP (overlay)
        # PRC
        "chn_fx":      ("EXCHUS",            "1990-01-01"),  # CNY/USD
        "chn_exports": ("XTEXVA01CNM667S",   "1992-01-01"),  # Merchandise exports
        "chn_ca_gdp":  ("CHNBCAGDPBP6PT",   "1997-01-01"),  # CA % GDP
        "chn_reserves":("TRESEGCNM052N",     "1990-01-01"),  # Reserves $M
        "chn_cpi":     ("CHNCPIALLMINMEI",   "1993-01-01"),  # CPI
        "chn_gdp":     ("MKTGDPCNA646NWDB",  "1990-01-01"),  # GDP USD
    }

    print("Pulling FRED series...")
    for name, (sid, start) in series_map.items():
        try:
            s = fred.get_series(sid, observation_start=start)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            data[name] = s
            print(f"  OK {sid} ({name}): {s.index[0].date()} to "
                  f"{s.index[-1].date()}, {len(s)} obs")
        except Exception as e:
            print(f"  FAIL {sid} ({name}): {e}")

    return data

# ── CELL 6: BUILD PANELS ─────────────────────────────────────────────────────

def build_japan_panel(data):
    """
    Japan annual panel 1976-1995.
    Variables: ca_gdp, cons_pct, savings_pct, fx_usd,
               exports_idx, real_wage, yield_10y
    """
    print("\n=== JAPAN PANEL ===")

    d = {}

    # CA from hardcoded
    d["ca_gdp"]     = make_annual_series(JAPAN_CA,   "ca_gdp")
    d["cons_pct"]   = make_annual_series(JAPAN_CONS, "cons_pct")
    d["savings_pct"]= make_annual_series(JAPAN_SAV,  "savings_pct")

    # FX annual mean
    if "jpn_fx" in data:
        d["fx_usd"] = data["jpn_fx"].resample("AS").mean()

    # Export index -- net trade value, annual sum
    if "jpn_exports" in data:
        exp = data["jpn_exports"].resample("AS").sum()
        # Normalize to index (1985=100)
        base = exp.loc["1985-01-01"] if "1985-01-01" in exp.index else exp.iloc[9]
        if base != 0:
            exp_idx = exp / base * 100
        else:
            exp_idx = exp
        exp_idx.name = "exports_idx"
        d["exports_idx"] = exp_idx

    # Real wage: nominal wage / CPI
    wage_key = "jpn_wage" if "jpn_wage" in data else "jpn_wage2"
    if wage_key in data and "jpn_cpi" in data:
        wage_ann = data[wage_key].resample("AS").mean()
        cpi_ann  = data["jpn_cpi"].resample("AS").mean()
        real_wage = (wage_ann / cpi_ann * 100)
        real_wage.name = "real_wage"
        d["real_wage"] = real_wage
        print(f"  Real wage constructed from {wage_key}")

    # 10yr JGB yield
    if "jpn_yield" in data:
        d["yield_10y"] = data["jpn_yield"].resample("AS").mean()

    # Demographics
    d["wap"]       = make_annual_series(JPN_WAP, "wap")       # working age pop
    d["dep_ratio"] = make_annual_series(JPN_DEP, "dep_ratio") # dependency ratio
    # WAP annual change (millions)
    wap_s = d["wap"]
    d["wap_chg"] = wap_s.diff()
    d["wap_chg"].name = "wap_chg"
    print(f"  Japan demographics loaded: WAP peaked at "
          f"{int(d['wap'].max())}M in "
          f"{d['wap'].idxmax().year}")

    # Assemble panel
    panel = pd.DataFrame(d)
    panel.index = pd.to_datetime(panel.index).tz_localize(None)

    # Clip to 1976-1995
    panel = panel[(panel.index >= "1976-01-01") &
                  (panel.index <= "1995-12-31")]

    # Drop rows missing core variables
    core = [c for c in ["ca_gdp","cons_pct","fx_usd","exports_idx"]
            if c in panel.columns]
    panel = panel.dropna(subset=core)

    print(f"  Panel: {panel.index[0].year}-{panel.index[-1].year}, "
          f"{len(panel)} obs")
    for col in panel.columns:
        print(f"    {col}: {panel[col].min():.2f} to {panel[col].max():.2f}")

    return panel


def build_prc_panel(data):
    """
    PRC annual panel 1997-2024.
    Variables: ca_gdp, cons_pct, savings_pct, fx_usd,
               exports_idx, reserves_b, routing_gap
    """
    print("\n=== PRC PANEL ===")

    d = {}

    d["cons_pct"]    = make_annual_series(PRC_CONS, "cons_pct")
    d["savings_pct"] = make_annual_series(PRC_SAV,  "savings_pct")

    if "chn_ca_gdp" in data:
        d["ca_gdp"] = data["chn_ca_gdp"].resample("AS").mean()

    if "chn_fx" in data:
        d["fx_usd"] = data["chn_fx"].resample("AS").mean()

    if "chn_exports" in data:
        exp = data["chn_exports"].resample("AS").sum()
        base_yr = "2000-01-01"
        base = exp.loc[base_yr] if base_yr in exp.index else exp.iloc[0]
        if base != 0:
            exp_idx = exp / base * 100
        else:
            exp_idx = exp
        exp_idx.name = "exports_idx"
        d["exports_idx"] = exp_idx

    if "chn_reserves" in data:
        res = data["chn_reserves"].resample("AS").last() / 1000  # to $B
        d["reserves_b"] = res

    # Demographics
    d["wap"]       = make_annual_series(PRC_WAP, "wap")
    d["dep_ratio"] = make_annual_series(PRC_DEP, "dep_ratio")
    wap_s = d["wap"]
    d["wap_chg"] = wap_s.diff()
    d["wap_chg"].name = "wap_chg"
    print(f"  PRC demographics loaded: WAP peaked at "
          f"{int(d['wap'].max())}M in "
          f"{d['wap'].idxmax().year}")
    print(f"  Current WAP decline: "
          f"{int(d['wap_chg'].iloc[-1])}M per year")

    # Routing gap: CA surplus minus reserve accumulation
    if "ca_gdp" in d and "reserves_b" in d and "chn_gdp" in data:
        gdp_b    = data["chn_gdp"].resample("AS").last() / 1e9
        ca_b     = d["ca_gdp"] / 100 * gdp_b
        res_chg  = d["reserves_b"].diff()
        d["routing_gap"] = (ca_b - res_chg)
        d["routing_gap"].name = "routing_gap"
        print("  Routing gap computed")

    panel = pd.DataFrame(d)
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    panel = panel[(panel.index >= "1997-01-01") &
                  (panel.index <= "2024-12-31")]

    core = [c for c in ["ca_gdp","cons_pct","fx_usd"]
            if c in panel.columns]
    panel = panel.dropna(subset=core)

    print(f"  Panel: {panel.index[0].year}-{panel.index[-1].year}, "
          f"{len(panel)} obs")
    for col in panel.columns:
        print(f"    {col}: {panel[col].min():.2f} to {panel[col].max():.2f}")

    return panel

# ── CELL 7: ADF ──────────────────────────────────────────────────────────────

def run_adf(panel, name):
    print(f"\n{'='*55}")
    print(f"ADF -- {name}")
    print("="*55)
    for col in panel.columns:
        s = panel[col].dropna()
        if len(s) < 8: continue
        try:
            lev = adfuller(s, autolag="AIC")
            d1  = adfuller(s.diff().dropna(), autolag="AIC")
            stat_l = "I(0)" if lev[1]<0.05 else "unit root"
            stat_d = "I(1)" if d1[1]<0.05 else "non-stationary"
            print(f"  {col:20s}: level p={lev[1]:.3f} ({stat_l}), "
                  f"diff p={d1[1]:.3f} ({stat_d})")
        except:
            print(f"  {col}: ADF failed")

# ── CELL 8: BIVARIATE GRANGER ────────────────────────────────────────────────

def run_bivariate_granger(panel, panel_name, hypotheses):
    print(f"\n{'='*65}")
    print(f"BIVARIATE GRANGER -- {panel_name} ({len(panel)} obs)")
    print("="*65)
    results = []

    for hyp_name, x_var, y_var, y_label in hypotheses:
        if x_var not in panel.columns or y_var not in panel.columns:
            print(f"\n  SKIP {hyp_name}: missing {x_var} or {y_var}")
            continue

        df = panel[[x_var, y_var]].dropna()
        df_use = df.diff().dropna()

        if len(df_use) < 8:
            print(f"\n  SKIP {hyp_name}: only {len(df_use)} obs")
            continue

        max_lag = min(3, len(df_use)//4)
        lags    = [l for l in [1,2,3] if l <= max_lag] or [1]

        try:
            t_xy = grangercausalitytests(df_use[[y_var,x_var]],
                                         maxlag=max(lags), verbose=False)
            pv_xy = {l: t_xy[l][0]["ssr_ftest"][1] for l in lags}
            min_p_xy = min(pv_xy.values())
            best_xy  = min(pv_xy, key=pv_xy.get)
        except: min_p_xy = 1.0; best_xy = 0

        try:
            t_yx = grangercausalitytests(df_use[[x_var,y_var]],
                                         maxlag=max(lags), verbose=False)
            pv_yx = {l: t_yx[l][0]["ssr_ftest"][1] for l in lags}
            min_p_yx = min(pv_yx.values())
            best_yx  = min(pv_yx, key=pv_yx.get)
        except: min_p_yx = 1.0; best_yx = 0

        sig_xy = min_p_xy < 0.10
        sig_yx = min_p_yx < 0.10

        if   sig_xy and not sig_yx: verdict = "CONFIRMED"
        elif sig_xy and sig_yx:     verdict = "BIDIRECTIONAL"
        elif not sig_xy and sig_yx: verdict = "REVERSED"
        else:                       verdict = "NULL"

        st = lambda p: "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
        print(f"\n  {hyp_name} [n={len(df_use)}, lag<={max_lag}]")
        print(f"    {x_var} -> {y_label}: p={min_p_xy:.4f}{st(min_p_xy)} "
              f"(lag {best_xy}yr)")
        print(f"    {y_label} -> {x_var}: p={min_p_yx:.4f}{st(min_p_yx)} "
              f"(lag {best_yx}yr)")
        print(f"    Verdict: {verdict}")

        results.append({
            "panel": panel_name, "hypothesis": hyp_name,
            "x_to_y_p": min_p_xy, "y_to_x_p": min_p_yx,
            "verdict": verdict, "obs": len(df_use),
        })

    return results

# ── CELL 9: MEDIATION VAR -- THE REBALANCING TRAP ────────────────────────────

def run_trap_mediation(panel, panel_name, trap_vars, label):
    """
    Test the rebalancing trap via sequential Granger mediation.

    Trap chain: consumption_rise -> FX_appreciation ->
                export_fall -> wage_fall -> consumption_reversal

    For each step test whether the step is significant.
    If all steps significant: trap confirmed.
    If step breaks: identify where the mechanism fails.

    Also run 3-var VAR and extract IRF: shock to consumption
    should show U-shape if trap is operating.
    """
    print(f"\n{'='*65}")
    print(f"REBALANCING TRAP MEDIATION -- {panel_name}")
    print(f"Chain: {' -> '.join(trap_vars)}")
    print("="*65)

    # Check all variables present
    available = [v for v in trap_vars if v in panel.columns]
    if len(available) < 3:
        print(f"  Only {len(available)} of {len(trap_vars)} variables available: {available}")
        print("  Need at least 3 for VAR -- skipping")
        return None, None

    df = panel[available].dropna()
    df_use = df.diff().dropna()

    if len(df_use) < 10:
        print(f"  Only {len(df_use)} obs after differencing -- too few for VAR")
        return None, None

    print(f"\n  Variables: {available}")
    print(f"  Obs after differencing: {len(df_use)}")

    # Sequential Granger steps
    print("\n  Sequential mediation steps:")
    step_results = []
    for i in range(len(available)-1):
        x_var = available[i]
        y_var = available[i+1]
        df_step = df_use[[x_var, y_var]].dropna()
        max_lag = min(2, len(df_step)//4)
        if max_lag < 1: continue
        try:
            t = grangercausalitytests(df_step[[y_var,x_var]],
                                      maxlag=max_lag, verbose=False)
            p = min(t[l][0]["ssr_ftest"][1] for l in range(1,max_lag+1))
            st = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "n.s."
            print(f"    Step {i+1}: {x_var} -> {y_var}: p={p:.4f} {st}")
            step_results.append({"step": i+1, "from": x_var,
                                 "to": y_var, "p": p})
        except Exception as e:
            print(f"    Step {i+1}: {x_var} -> {y_var}: FAILED ({e})")

    # Trap verdict
    significant_steps = [s for s in step_results if s["p"] < 0.10]
    total_steps = len(step_results)
    if total_steps > 0:
        if len(significant_steps) == total_steps:
            trap_verdict = "TRAP CONFIRMED -- all chain steps significant"
        elif len(significant_steps) >= total_steps * 0.6:
            trap_verdict = f"TRAP PARTIAL -- {len(significant_steps)}/{total_steps} steps significant"
        else:
            trap_verdict = f"TRAP WEAK -- only {len(significant_steps)}/{total_steps} steps significant"
        print(f"\n  Trap verdict: {trap_verdict}")

    # VAR and IRF
    print(f"\n  Running VAR IRF (shock to {available[0]})...")
    irf_data = None
    try:
        max_lag_var = min(2, (len(df_use)-1) // len(available))
        if max_lag_var < 1: max_lag_var = 1
        model  = VAR(df_use[available])
        fitted = model.fit(max_lag_var)
        irf    = fitted.irf(periods=8)
        irf_data = irf
        print(f"  VAR fitted: lag={max_lag_var}, {len(available)} vars")

        # Print IRF of consumption shock on each variable
        cons_idx = available.index(available[0])
        print(f"\n  IRF: {available[0]} shock -> downstream:")
        for j, var in enumerate(available[1:], 1):
            response = irf.irfs[:, cons_idx, j]
            direction = "UP" if response[2] > 0 else "DOWN"
            reversal  = "REVERSAL" if (response[0] > 0 and response[-1] < 0) else \
                        "NO REVERSAL"
            print(f"    -> {var}: peak={response.max():.4f}, "
                  f"trough={response.min():.4f}, direction={direction}, {reversal}")

    except Exception as e:
        print(f"  VAR failed: {e}")

    return step_results, irf_data

# ── CELL 10: CHARTS ──────────────────────────────────────────────────────────

def plot_overview(jpn, prc):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor("#03060d")
    fig.suptitle(
        "The Surplus Side of the Circuit -- Japan 1976-1995 | PRC 1997-2024\n"
        "Mirror of AmeriCo: opposite sign, same structural logic",
        color="#e2e8f0", fontsize=12
    )

    def sp(ax, series, title, color, ylabel=""):
        ax.set_facecolor("#03060d")
        if series is not None and len(series.dropna()) > 0:
            ax.plot(series.dropna(), color=color, linewidth=2)
        ax.set_title(title, color="#e2e8f0", fontsize=9)
        ax.set_ylabel(ylabel, color="#4a5568", fontsize=7)
        ax.tick_params(colors="#4a5568", labelsize=7)
        for s in ax.spines.values(): s.set_color("#1e2d3a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Row 1: Japan
    cols_j = ["ca_gdp","cons_pct","exports_idx","fx_usd"]
    titles_j = ["CA Surplus % GDP","Consumption % GDP",
                "Export Index (1985=100)","JPY/USD (higher=weaker)"]
    colors_j = ["#22d3ee","#f59e0b","#ef4444","#a78bfa"]
    for ax, col, title, color in zip(axes[0], cols_j, titles_j, colors_j):
        series = jpn[col] if jpn is not None and col in jpn.columns else None
        sp(ax, series, f"Japan {title}", color)
        if series is not None:
            ax.axvline(pd.Timestamp("1985-01-01"), color="#22c55e",
                      linewidth=0.9, linestyle="--", alpha=0.7, label="Plaza")
            ax.legend(fontsize=6, labelcolor="#64748b",
                     facecolor="#03060d", edgecolor="#1e2d3a")

    # Row 2: PRC
    cols_c = ["ca_gdp","cons_pct","exports_idx","fx_usd"]
    titles_c = ["CA Surplus % GDP","Consumption % GDP",
                "Export Index (2000=100)","CNY/USD (managed)"]
    colors_c = ["#22d3ee","#f59e0b","#ef4444","#a78bfa"]
    for ax, col, title, color in zip(axes[1], cols_c, titles_c, colors_c):
        series = prc[col] if prc is not None and col in prc.columns else None
        sp(ax, series, f"PRC {title}", color)
        if series is not None:
            for yr, lbl, col2 in [("2001-01-01","WTO","#22c55e"),
                                   ("2015-01-01","Devaluation","#f87171")]:
                ax.axvline(pd.Timestamp(yr), color=col2, linewidth=0.8,
                          linestyle="--", alpha=0.7, label=lbl)
            ax.legend(fontsize=6, labelcolor="#64748b",
                     facecolor="#03060d", edgecolor="#1e2d3a")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "kappa_J1_surplus_overview_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")


def plot_trap_irf(irf_data, available, panel_name, save_name):
    """
    Plot IRF from consumption shock -- the rebalancing trap visualization.
    If trap operates: consumption shock -> FX appreciates -> exports fall -> reversal
    """
    if irf_data is None:
        print(f"No IRF data for {panel_name}")
        return

    n_vars = len(available)
    fig, axes = plt.subplots(1, n_vars-1, figsize=(4*(n_vars-1), 5))
    if n_vars == 2: axes = [axes]
    fig.patch.set_facecolor("#03060d")
    fig.suptitle(
        f"Rebalancing Trap IRF -- {panel_name}\n"
        f"Shock to {available[0]} -- downstream responses",
        color="#e2e8f0", fontsize=11
    )

    colors = ["#f59e0b","#ef4444","#a78bfa","#22d3ee","#22c55e"]
    cons_idx = 0  # shock is always first variable

    for i, (ax, var) in enumerate(zip(axes, available[1:])):
        ax.set_facecolor("#03060d")
        response = irf_data.irfs[:, cons_idx, i+1]
        periods  = range(len(response))

        # Color by direction
        color = colors[i % len(colors)]
        ax.plot(periods, response, color=color, linewidth=2.5)
        ax.axhline(0, color="#334155", linewidth=0.8)

        # Fill positive/negative
        ax.fill_between(periods, response, 0,
                        where=[r > 0 for r in response],
                        alpha=0.15, color="#22c55e")
        ax.fill_between(periods, response, 0,
                        where=[r <= 0 for r in response],
                        alpha=0.15, color="#ef4444")

        # Annotate reversal if present
        if response[0] > 0 and response[-1] < 0:
            ax.annotate("REVERSAL\n(trap operating)",
                       xy=(len(response)-1, response[-1]),
                       color="#ef4444", fontsize=7,
                       xytext=(len(response)//2, response.min()*0.7),
                       arrowprops=dict(arrowstyle="->",
                                      color="#ef4444", lw=0.8))

        ax.set_title(f"-> {var}", color="#e2e8f0", fontsize=10)
        ax.set_xlabel("Years after consumption shock",
                     color="#4a5568", fontsize=8)
        ax.tick_params(colors="#4a5568", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#1e2d3a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")


def plot_suppression_scatter(jpn, prc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#03060d")
    fig.suptitle(
        "CA Surplus vs Consumption Suppression\n"
        "More surplus = less domestic consumption -- the suppression identity",
        color="#e2e8f0", fontsize=11
    )

    for ax, panel, title, color in [
        (axes[0], jpn, "Japan 1976-1995", "#22d3ee"),
        (axes[1], prc, "PRC 1997-2024",   "#f59e0b"),
    ]:
        ax.set_facecolor("#03060d")
        if panel is None or "ca_gdp" not in panel.columns \
                         or "cons_pct" not in panel.columns:
            ax.set_title(f"{title} -- no data", color="#e2e8f0")
            continue

        df = panel[["ca_gdp","cons_pct"]].dropna()
        # Color by decade
        decade_colors = []
        for t in df.index:
            if t.year < 1986: decade_colors.append("#22c55e")
            elif t.year < 1991: decade_colors.append("#f59e0b")
            else: decade_colors.append("#ef4444")

        ax.scatter(df["ca_gdp"], df["cons_pct"],
                  c=decade_colors, alpha=0.8, s=50, edgecolors="none")

        # Label years
        for idx, row in df.iterrows():
            ax.annotate(str(idx.year),
                       (row["ca_gdp"], row["cons_pct"]),
                       fontsize=6, color="#64748b",
                       xytext=(2, 2), textcoords="offset points")

        # Trend
        z = np.polyfit(df["ca_gdp"], df["cons_pct"], 1)
        xline = np.linspace(df["ca_gdp"].min(), df["ca_gdp"].max(), 100)
        ax.plot(xline, np.poly1d(z)(xline), color=color,
               linewidth=1.5, linestyle="--", alpha=0.8)

        corr = np.corrcoef(df["ca_gdp"], df["cons_pct"])[0,1]
        ax.set_title(f"{title}\nr={corr:.3f}", color="#e2e8f0", fontsize=10)
        ax.set_xlabel("CA Surplus % GDP", color="#4a5568", fontsize=8)
        ax.set_ylabel("Household Consumption % GDP",
                     color="#4a5568", fontsize=8)
        ax.tick_params(colors="#4a5568", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#1e2d3a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "kappa_J2_suppression_scatter_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")


def plot_granger_summary(all_results):
    if not all_results: return
    df = pd.DataFrame(all_results)

    fig, ax = plt.subplots(figsize=(13, max(5, len(df)*0.45)))
    fig.patch.set_facecolor("#03060d")
    ax.set_facecolor("#03060d")

    verdict_colors = {
        "CONFIRMED":    "#22c55e",
        "BIDIRECTIONAL":"#f59e0b",
        "REVERSED":     "#f87171",
        "NULL":         "#334155",
    }

    for i, (_, row) in enumerate(df.iterrows()):
        color = verdict_colors.get(row["verdict"], "#334155")
        ax.barh(i-0.2, -np.log10(max(row["x_to_y_p"], 0.001)),
               height=0.35, color=color, alpha=0.9)
        ax.barh(i+0.2, -np.log10(max(row["y_to_x_p"], 0.001)),
               height=0.35, color=color, alpha=0.35)

    for p_val, lbl, col in [(0.10,"p=0.10","#ffffff"),
                              (0.05,"p=0.05","#f59e0b"),
                              (0.01,"p=0.01","#22c55e")]:
        ax.axvline(-np.log10(p_val), color=col, linewidth=0.8,
                  linestyle="--", alpha=0.5, label=lbl)

    labels = [f"{r['panel'][:18]} | {r['hypothesis'][:32]}"
              for _, r in df.iterrows()]
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, color="#94a3b8", fontsize=7)
    ax.set_xlabel("-log10(p-value)", color="#4a5568", fontsize=9)
    ax.set_title("Granger Causality -- Surplus Side\n"
                "Japan 1976-1995 | PRC 1997-2024",
                color="#e2e8f0", fontsize=10)
    ax.tick_params(colors="#4a5568", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#1e2d3a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, labelcolor="#64748b",
             facecolor="#03060d", edgecolor="#1e2d3a")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "kappa_J4_granger_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")

# ── CELL 11: HYPOTHESES ──────────────────────────────────────────────────────

# Japan bivariate
JAPAN_HYP = [
    ("H1J: CA surplus -> consumption suppression",
     "ca_gdp",    "cons_pct",    "Consumption % GDP"),
    ("H2J: FX undervaluation -> CA surplus",
     "fx_usd",    "ca_gdp",      "CA % GDP"),
    ("H3J: FX appreciation -> export decline",
     "fx_usd",    "exports_idx", "Export Index"),
    ("H4J: Export decline -> consumption reversal",
     "exports_idx","cons_pct",   "Consumption % GDP"),
    ("H5J: CA surplus -> savings accumulation",
     "ca_gdp",    "savings_pct", "Savings % GDP"),
    ("H6J: Wage growth -> consumption",
     "real_wage", "cons_pct",    "Consumption % GDP"),
]

# PRC bivariate
PRC_HYP = [
    ("H1C: CA surplus -> consumption suppression",
     "ca_gdp",      "cons_pct",    "Consumption % GDP"),
    ("H2C: FX management -> CA surplus",
     "fx_usd",      "ca_gdp",      "CA % GDP"),
    ("H3C: Export growth -> consumption",
     "exports_idx", "cons_pct",    "Consumption % GDP"),
    ("H4C: FX -> exports",
     "fx_usd",      "exports_idx", "Export Index"),
    ("H5C: Reserve accumulation -> consumption suppression",
     "reserves_b",  "cons_pct",    "Consumption % GDP"),
    ("H6C: Routing gap -> consumption suppression",
     "routing_gap", "cons_pct",    "Consumption % GDP"),
    ("H7C: CA surplus -> savings accumulation",
     "ca_gdp",      "savings_pct", "Savings % GDP"),
]

# Add demographic hypotheses
PRC_HYP_DEMO = [
    ("H8C: WAP decline -> consumption share rising",
     "wap_chg",  "cons_pct",    "Consumption % GDP"),
    ("H9C: Dependency ratio -> savings rate falling",
     "dep_ratio","savings_pct", "Savings % GDP"),
    ("H10C: WAP decline -> CA surplus compression",
     "wap_chg",  "ca_gdp",      "CA % GDP"),
    ("H11C: Demographic forcing -> fiscal rebalancing vs productive rebalancing",
     "dep_ratio","ca_gdp",      "CA % GDP"),
]

JPN_HYP_DEMO = [
    ("H7J: WAP growth -> consumption rising",
     "wap_chg",  "cons_pct",    "Consumption % GDP"),
    ("H8J: Dependency ratio -> savings rate",
     "dep_ratio","savings_pct", "Savings % GDP"),
    ("H9J: WAP growth -> CA surplus",
     "wap_chg",  "ca_gdp",      "CA % GDP"),
]

# Rebalancing trap chains
JAPAN_TRAP = ["cons_pct", "fx_usd", "exports_idx"]
# consumption rise -> FX appreciates (note: fx_usd = JPY per dollar,
# so rising cons -> falling fx_usd = yen appreciation)
# -> exports fall

PRC_TRAP   = ["cons_pct", "fx_usd", "exports_idx", "ca_gdp"]
# consumption rise -> currency appreciates (fx = CNY/USD falls)
# -> exports fall -> CA surplus compresses -> reinforces consumption?

def plot_demographics(jpn, prc):
    """Chart: demographic forcing -- WAP and dependency ratio for both countries.
    Shows Japan 30 years ahead of PRC on the same demographic path."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#03060d")
    fig.suptitle(
        "Demographic Forcing -- The Non-Deferrable Constraint\n"
        "Japan 1976-1995 (left) | PRC 1997-2024 (right)\n"
        "PRC is 30 years behind Japan on the same demographic path",
        color="#e2e8f0", fontsize=11
    )

    def sp(ax, series, title, color, ylabel=""):
        ax.set_facecolor("#03060d")
        if series is not None and len(series.dropna()) > 0:
            ax.plot(series.dropna(), color=color, linewidth=2)
        ax.set_title(title, color="#e2e8f0", fontsize=9)
        ax.set_ylabel(ylabel, color="#4a5568", fontsize=7)
        ax.tick_params(colors="#4a5568", labelsize=7)
        for s in ax.spines.values(): s.set_color("#1e2d3a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Japan WAP and dependency
    if jpn is not None:
        if "wap" in jpn.columns:
            sp(axes[0,0], jpn["wap"], "Japan WAP (millions) 1976-1995",
               "#22d3ee", "Millions")
            axes[0,0].axvline(pd.Timestamp("1990-01-01"),
                            color="#f59e0b", linewidth=0.8,
                            linestyle="--", alpha=0.7, label="Bubble peak")
            axes[0,0].legend(fontsize=7, labelcolor="#64748b",
                           facecolor="#03060d", edgecolor="#1e2d3a")
        if "dep_ratio" in jpn.columns:
            sp(axes[1,0], jpn["dep_ratio"],
               "Japan Old-Age Dependency Ratio 1976-1995",
               "#f59e0b", "Ratio (65+/15-64 %)")

    # PRC WAP and dependency
    if prc is not None:
        if "wap" in prc.columns:
            sp(axes[0,1], prc["wap"], "PRC WAP (millions) 1997-2024",
               "#22d3ee", "Millions")
            axes[0,1].axvline(pd.Timestamp("2011-01-01"),
                            color="#ef4444", linewidth=0.8,
                            linestyle="--", alpha=0.8, label="WAP peak 2011")
            axes[0,1].legend(fontsize=7, labelcolor="#64748b",
                           facecolor="#03060d", edgecolor="#1e2d3a")
        if "dep_ratio" in prc.columns:
            sp(axes[1,1], prc["dep_ratio"],
               "PRC Old-Age Dependency Ratio 1997-2024",
               "#f59e0b", "Ratio (65+/15-64 %)")

    # Annotation: the window
    for ax in [axes[0,1], axes[1,1]]:
        ax.axvspan(pd.Timestamp("2025-01-01"),
                  pd.Timestamp("2035-01-01"),
                  alpha=0.08, color="#22c55e",
                  label="10yr rebalancing window")
        ax.legend(fontsize=6, labelcolor="#64748b",
                 facecolor="#03060d", edgecolor="#1e2d3a")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "kappa_J6_demographics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")


# ── CELL 12: MAIN ────────────────────────────────────────────────────────────

def main():
    print("kappa JapanCo/PRC Surplus Analysis v2")
    print("Rebalancing trap: consumption -> FX -> exports -> reversal")
    print("="*60)

    print("\n[1] Pulling FRED data...")
    data = pull_all_data(FRED_API_KEY)

    print("\n[2] Building panels...")
    jpn = build_japan_panel(data)
    prc = build_prc_panel(data)

    print("\n[3] ADF unit root tests...")
    if jpn is not None and len(jpn) > 0: run_adf(jpn, "Japan 1976-1995")
    if prc is not None and len(prc) > 0: run_adf(prc, "PRC 1997-2024")

    print("\n[4] Bivariate Granger tests...")
    all_results = []
    if jpn is not None and len(jpn) >= 8:
        all_results.extend(
            run_bivariate_granger(jpn, "Japan 1976-1995", JAPAN_HYP)
        )
        print("\n  -- Japan demographic hypotheses --")
        all_results.extend(
            run_bivariate_granger(jpn, "Japan 1976-1995 (demo)",
                                  JPN_HYP_DEMO)
        )
    if prc is not None and len(prc) >= 8:
        all_results.extend(
            run_bivariate_granger(prc, "PRC 1997-2024", PRC_HYP)
        )
        print("\n  -- PRC demographic hypotheses --")
        all_results.extend(
            run_bivariate_granger(prc, "PRC 1997-2024 (demo)",
                                  PRC_HYP_DEMO)
        )

    print("\n[5] Rebalancing trap mediation + VAR IRF...")
    jpn_steps, jpn_irf = None, None
    prc_steps, prc_irf = None, None

    if jpn is not None and len(jpn) >= 10:
        jpn_steps, jpn_irf = run_trap_mediation(
            jpn, "Japan 1976-1995", JAPAN_TRAP,
            "Japan rebalancing trap"
        )
    if prc is not None and len(prc) >= 10:
        prc_steps, prc_irf = run_trap_mediation(
            prc, "PRC 1997-2024", PRC_TRAP,
            "PRC rebalancing trap"
        )

    print("\n[6] Generating charts...")
    plot_overview(jpn, prc)
    plot_suppression_scatter(jpn, prc)

    if jpn is not None and jpn_irf is not None:
        avail_j = [v for v in JAPAN_TRAP if v in jpn.columns]
        plot_trap_irf(jpn_irf, avail_j, "Japan 1976-1995",
                     "kappa_J3_japan_trap_irf.png")

    if prc is not None and prc_irf is not None:
        avail_c = [v for v in PRC_TRAP if v in prc.columns]
        plot_trap_irf(prc_irf, avail_c, "PRC 1997-2024",
                     "kappa_J5_prc_trap_irf.png")

    plot_granger_summary(all_results)

    # Demographic chart
    plot_demographics(jpn, prc)

    # Print summary table
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{'='*80}")
        print("GRANGER SUMMARY -- SURPLUS SIDE")
        print("="*80)
        print(f"{'Panel':<22} {'Hypothesis':<38} {'X->Y':>9} "
              f"{'Y->X':>9} {'Obs':>4} {'Verdict'}")
        print("-"*80)
        for _, r in df.iterrows():
            st = lambda p: "***" if p<0.01 else "**" if p<0.05 \
                           else "*" if p<0.10 else ""
            xp = f"{r['x_to_y_p']:.4f}{st(r['x_to_y_p'])}"
            yp = f"{r['y_to_x_p']:.4f}{st(r['y_to_x_p'])}"
            print(f"{r['panel']:<22} {r['hypothesis']:<38} "
                  f"{xp:>9} {yp:>9} {int(r['obs']):>4} {r['verdict']}")
        print("="*80)

        df.to_csv(
            os.path.join(SAVE_DIR, "kappa_surplus_granger_v2.csv"),
            index=False
        )

    if jpn is not None:
        jpn.to_csv(os.path.join(SAVE_DIR, "kappa_japan_panel_v2.csv"))
    if prc is not None:
        prc.to_csv(os.path.join(SAVE_DIR, "kappa_prc_panel_v2.csv"))

    print(f"\nAll outputs saved to {SAVE_DIR}")
    print("Charts: kappa_J1_v2 through kappa_J5")
    print("CSVs: kappa_surplus_granger_v2.csv, kappa_japan_panel_v2.csv, "
          "kappa_prc_panel_v2.csv")

    return jpn, prc, all_results

if __name__ == "__main__":
    jpn, prc, results = main()
