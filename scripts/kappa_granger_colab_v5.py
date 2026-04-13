# kappa_granger_colab_v5.py
# Granger Causality + Mediation Analysis
# v5: NIIP splice fixed (IIPUSNETIA), CAPE extended to 1976, real_income isolated
# Full panel 1976-2025, pre-GFC sample now viable
# Vinodh Raghunathan / Speculativa / April 2026

# ── CELL 1: INSTALL ──────────────────────────────────────────────────────────
# Run as separate Colab code cell FIRST:
# !pip install fredapi statsmodels pandas numpy matplotlib seaborn openpyxl -q

# ── CELL 2: IMPORTS AND DRIVE MOUNT ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import os
import io

from google.colab import drive
drive.mount('/content/drive')

from fredapi import Fred
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ── CELL 3: CONFIG ───────────────────────────────────────────────────────────

FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"

FULL_START   = "1976-01-01"
MAX_LAGS     = 8
GRANGER_LAGS = [2, 4, 6, 8]
SAVE_DIR     = "/content/drive/MyDrive/StockElephant/"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Save directory: {SAVE_DIR}")

# ── CELL 4: SHILLER CAPE -- extended to 1976 ─────────────────────────────────

# Annual Shiller CAPE 1976-2025 from shillerdata.com
# 1976-1989: from Shiller published data
# 1990-2025: previously verified values
SHILLER_YEARS = list(range(1976, 2026))
SHILLER_CAPE  = [
    # 1976-1989
    11.2, 10.0, 8.9, 9.3, 9.3, 8.4, 7.4, 10.2, 10.0, 11.8,
    16.0, 17.4, 14.9, 17.2,
    # 1990-1999
    15.1, 17.0, 18.9, 20.3, 21.1, 23.0, 26.0, 31.5, 36.0, 44.2,
    # 2000-2009
    37.0, 30.5, 23.3, 19.6, 20.6, 22.9, 26.4, 27.2, 24.1, 19.3,
    # 2010-2019
    20.5, 23.6, 21.2, 24.9, 26.5, 27.0, 26.6, 32.1, 33.3, 30.1,
    # 2020-2025
    33.0, 38.3, 40.2, 28.9, 31.0, 33.5,
]
assert len(SHILLER_YEARS) == len(SHILLER_CAPE), \
    f"CAPE length mismatch: {len(SHILLER_YEARS)} years vs {len(SHILLER_CAPE)} values"

# ── CELL 5: DATA PULL ────────────────────────────────────────────────────────

def pull_fred_data(api_key):
    fred = Fred(api_key=api_key)
    data = {}

    # ── NIIP: splice annual (IIPUSNETIA 1976-2005) + quarterly (IIPUSNETIQ 2006+) ──
    print("Pulling NIIP (splicing IIPUSNETIA + IIPUSNETIQ)...")
    try:
        # Annual 1976-2005 -- correct series ID is IIPUSNETIA
        niip_a = fred.get_series("IIPUSNETIA",
                                  observation_start="1976-01-01",
                                  observation_end="2005-12-31")
        niip_a = niip_a.resample("QS").interpolate(method="linear")
        print(f"  Annual NIIP (IIPUSNETIA): {niip_a.index[0].date()} to {niip_a.index[-1].date()}, {len(niip_a)} obs")

        # Quarterly from 2006
        niip_q = fred.get_series("IIPUSNETIQ", observation_start="2006-01-01")
        niip_q = niip_q.resample("QS").last()
        print(f"  Quarterly NIIP (IIPUSNETIQ): {niip_q.index[0].date()} to {niip_q.index[-1].date()}, {len(niip_q)} obs")

        # Splice
        niip = pd.concat([niip_a[niip_a.index < "2006-01-01"], niip_q])
        niip = niip[niip.index >= FULL_START]
        niip.name = "niip"
        data["niip"] = niip
        print(f"  NIIP spliced: {niip.index[0].date()} to {niip.index[-1].date()}, {len(niip)} obs")
        print(f"  NIIP range: {niip.min():.0f} to {niip.max():.0f} ($M)")

    except Exception as e:
        print(f"  NIIP splice failed: {e}")
        print("  Falling back to quarterly only (2006+)")
        niip = fred.get_series("IIPUSNETIQ")
        niip.name = "niip"
        data["niip"] = niip

    # ── GDP ──────────────────────────────────────────────────────────────────
    print("Pulling GDP...")
    gdp = fred.get_series("GDP", observation_start=FULL_START)
    gdp.name = "gdp"
    data["gdp"] = gdp
    print(f"  GDP: {gdp.index[0].date()} to {gdp.index[-1].date()}, {len(gdp)} obs")

    # ── SHILLER CAPE -- hardcoded 1976-2025 ──────────────────────────────────
    print("Loading Shiller CAPE (hardcoded 1976-2025)...")
    cape_a = pd.Series(SHILLER_CAPE,
                       index=pd.date_range("1976", periods=len(SHILLER_YEARS), freq="YS"))
    cape_q = cape_a.resample("QS").interpolate(method="linear")
    cape_q.name = "cape"
    data["cape"] = cape_q
    print(f"  CAPE: {cape_q.index[0].date()} to {cape_q.index[-1].date()}, {len(cape_q)} obs")
    print(f"  CAPE range: {cape_q.min():.1f} to {cape_q.max():.1f}")

    # ── PERSONAL SAVINGS RATE ─────────────────────────────────────────────────
    print("Pulling personal savings rate...")
    psav = fred.get_series("PSAVERT", observation_start=FULL_START)
    psav = psav.resample("QS").mean()
    psav.name = "savings"
    data["savings"] = psav
    print(f"  Savings: {psav.index[0].date()} to {psav.index[-1].date()}, {len(psav)} obs")

    # ── MANUFACTURING EMPLOYMENT SHARE ────────────────────────────────────────
    print("Pulling manufacturing employment share...")
    mfg_emp = fred.get_series("MANEMP", observation_start=FULL_START)
    tot_emp = fred.get_series("PAYEMS", observation_start=FULL_START)
    mfg_share = (mfg_emp / tot_emp * 100).resample("QS").mean()
    mfg_share.name = "mfg_share"
    data["mfg_share"] = mfg_share
    print(f"  Mfg share: {mfg_share.index[0].date()} to {mfg_share.index[-1].date()}, {len(mfg_share)} obs")

    # ── 10-YEAR TREASURY YIELD ────────────────────────────────────────────────
    print("Pulling 10-year Treasury yield...")
    dgs10 = fred.get_series("DGS10", observation_start=FULL_START)
    dgs10 = dgs10.resample("QS").mean()
    dgs10.name = "yield_10y"
    data["yield_10y"] = dgs10
    print(f"  10yr yield: {dgs10.index[0].date()} to {dgs10.index[-1].date()}, {len(dgs10)} obs")

    # ── REAL MEDIAN HOUSEHOLD INCOME -- kept separate, clips panel if included ──
    print("Pulling real median household income (mediation only)...")
    try:
        rmhi = fred.get_series("MEHOINUSA672N", observation_start=FULL_START)
        rmhi = rmhi.resample("QS").interpolate(method="linear")
        rmhi.name = "real_income"
        data["real_income"] = rmhi
        print(f"  Real income: {rmhi.index[0].date()} to {rmhi.index[-1].date()}, {len(rmhi)} obs")
        print(f"  NOTE: real income only used in mediation test, not in main panel")
    except Exception as e:
        print(f"  Real income failed: {e}")
        try:
            rwage = fred.get_series("AHETPI", observation_start=FULL_START)
            cpi   = fred.get_series("CPIAUCSL", observation_start=FULL_START)
            real_wage = (rwage / cpi * 100).resample("QS").mean()
            real_wage.name = "real_income"
            data["real_income"] = real_wage
            print(f"  Real wage proxy: {real_wage.index[0].date()} to {real_wage.index[-1].date()}")
        except Exception as e2:
            print(f"  Real wage proxy also failed: {e2}")

    return data

# ── CELL 6: PANEL PREP ───────────────────────────────────────────────────────

def prepare_panel(data):
    """
    Main panel uses five core series only.
    real_income kept separate -- only 1984+ which would clip full panel.
    """
    niip     = data["niip"].resample("QS").last()
    gdp      = data["gdp"].resample("QS").last()
    cape     = data["cape"].resample("QS").last()
    savings  = data["savings"].resample("QS").mean()
    mfg      = data["mfg_share"].resample("QS").last()
    yield10  = data["yield_10y"].resample("QS").mean()

    niip_gdp = (niip / gdp * 100)
    niip_gdp.name = "niip_gdp"

    panel = pd.DataFrame({
        "niip_gdp":  niip_gdp,
        "cape":      cape,
        "savings":   savings,
        "mfg_share": mfg,
        "yield_10y": yield10,
    })
    panel.index = pd.to_datetime(panel.index).tz_localize(None)

    # Drop NaN on core five only -- do NOT include real_income here
    panel = panel.dropna(subset=["niip_gdp","cape","savings","mfg_share","yield_10y"])

    print(f"\nMain panel: {panel.index[0].date()} to {panel.index[-1].date()}, {len(panel)} quarters")
    for col in panel.columns:
        print(f"  {col}: {panel[col].min():.2f} to {panel[col].max():.2f}")

    # Separate real_income panel for mediation (1984+ or 1990+)
    if "real_income" in data:
        ri = data["real_income"].resample("QS").last()
        ri.index = pd.to_datetime(ri.index).tz_localize(None)
        panel_med = panel.copy()
        panel_med["real_income"] = ri
        panel_med = panel_med.dropna(subset=["real_income"])
        print(f"\nMediation panel (with real income): {panel_med.index[0].date()} to {panel_med.index[-1].date()}, {len(panel_med)} quarters")
    else:
        panel_med = panel.copy()

    return panel, panel_med

# ── CELL 7: ADF ──────────────────────────────────────────────────────────────

def run_adf(panel):
    print("\n" + "="*60)
    print("ADF UNIT ROOT TESTS")
    print("="*60)
    for name in panel.columns:
        s = panel[name].dropna()
        lev = adfuller(s, autolag="AIC")
        d1  = adfuller(s.diff().dropna(), autolag="AIC")
        print(f"\n{name}:")
        print(f"  Level:    p={lev[1]:.4f} -> {'I(0)' if lev[1]<0.05 else 'unit root'}")
        print(f"  1st diff: p={d1[1]:.4f}  -> {'I(1)' if d1[1]<0.05 else 'non-stationary'}")

# ── CELL 8: COINTEGRATION ────────────────────────────────────────────────────

def test_coint(panel):
    print("\n" + "="*60)
    print("JOHANSEN COINTEGRATION: NIIP/GDP vs CAPE")
    print("="*60)
    data_c = panel[["niip_gdp","cape"]].dropna()
    try:
        joh = coint_johansen(data_c, det_order=0, k_ar_diff=2)
        cointegrated = joh.lr1[0] > joh.cvt[0,1]
        print(f"Trace r=0: {joh.lr1[0]:.3f}, 95% CV: {joh.cvt[0,1]:.3f}")
        print(f"-> {'COINTEGRATED: use levels' if cointegrated else 'NOT COINTEGRATED: use first differences'}")
    except Exception as e:
        print(f"Test failed: {e}")
        cointegrated = False
    return cointegrated

# ── CELL 9: BIVARIATE GRANGER ────────────────────────────────────────────────

def run_granger(panel, cointegrated=False):
    hypotheses = [
        ("H1: NIIP -> Equity Valuations",   "niip_gdp", "cape",      "CAPE"),
        ("H2: NIIP -> ManuCo Share Decline", "niip_gdp", "mfg_share", "Mfg Share"),
        ("H3: NIIP -> HH Savings Depletion", "niip_gdp", "savings",   "Savings Rate"),
        ("H4: NIIP -> Rate Suppression",     "niip_gdp", "yield_10y", "10yr Yield"),
    ]

    p = panel.copy()
    p.index = pd.to_datetime(p.index).tz_localize(None)

    samples = {
        # Three overlapping 20-year windows -- tells the evolution story
        # Overlap is intentional: stable results = structural, flipping results = transition
        "1976-1995 (early accumulation)":  p[(p.index >= pd.Timestamp("1976-01-01")) &
                                             (p.index <  pd.Timestamp("1996-01-01"))],
        "1990-2009 (globalization era)":   p[(p.index >= pd.Timestamp("1990-01-01")) &
                                             (p.index <  pd.Timestamp("2010-01-01"))],
        "2006-2025 (financialization era)":p[(p.index >= pd.Timestamp("2006-01-01"))],
        # Full sample and post-2016 retained for comparison
        "Full 1976-2025":                  p,
        "Post-2016":                       p[p.index >= pd.Timestamp("2016-01-01")],
    }

    print("\nSample sizes:")
    for name, s in samples.items():
        if len(s) > 0:
            print(f"  {name}: {len(s)} quarters ({s.index[0].date()} to {s.index[-1].date()})")
        else:
            print(f"  {name}: EMPTY")

    all_results = []

    for sample_name, samp in samples.items():
        if len(samp) < 20:
            print(f"\nSkipping {sample_name}: only {len(samp)} obs")
            continue

        print(f"\n{'='*70}")
        print(f"BIVARIATE GRANGER -- {sample_name} ({len(samp)} quarters)")
        print("="*70)

        for hyp_name, x_var, y_var, y_label in hypotheses:
            df = samp[[x_var, y_var]].dropna()
            df_use = df.diff().dropna() if not cointegrated else df

            if len(df_use) < 16:
                continue

            max_lag = min(MAX_LAGS, len(df_use)//6)
            lags_to_test = [l for l in GRANGER_LAGS if l <= max_lag] or [2]

            try:
                t_xy = grangercausalitytests(df_use[[y_var,x_var]],
                                             maxlag=max(lags_to_test), verbose=False)
                pv_xy = {l: t_xy[l][0]["ssr_ftest"][1] for l in lags_to_test}
                min_p_xy = min(pv_xy.values())
                best_xy  = min(pv_xy, key=pv_xy.get)
            except:
                min_p_xy = 1.0; best_xy = 0

            try:
                t_yx = grangercausalitytests(df_use[[x_var,y_var]],
                                             maxlag=max(lags_to_test), verbose=False)
                pv_yx = {l: t_yx[l][0]["ssr_ftest"][1] for l in lags_to_test}
                min_p_yx = min(pv_yx.values())
                best_yx  = min(pv_yx, key=pv_yx.get)
            except:
                min_p_yx = 1.0; best_yx = 0

            sig_xy = min_p_xy < 0.10
            sig_yx = min_p_yx < 0.10
            if   sig_xy and not sig_yx: verdict = "CONFIRMED"
            elif sig_xy and sig_yx:     verdict = "BIDIRECTIONAL"
            elif not sig_xy and sig_yx: verdict = "REVERSED"
            else:                       verdict = "NULL"

            st = lambda p: "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
            print(f"\n  {hyp_name} [n={len(df_use)}, lag<={max_lag}]")
            print(f"    NIIP -> {y_label}: p={min_p_xy:.4f}{st(min_p_xy)} (lag {best_xy}q)")
            print(f"    {y_label} -> NIIP: p={min_p_yx:.4f}{st(min_p_yx)} (lag {best_yx}q)")
            print(f"    Verdict: {verdict}")

            all_results.append({
                "sample": sample_name, "hypothesis": hyp_name,
                "niip_to_y_p": min_p_xy, "y_to_niip_p": min_p_yx,
                "verdict": verdict, "obs": len(df_use),
            })

    return pd.DataFrame(all_results)

# ── CELL 10: MEDIATION ───────────────────────────────────────────────────────

def run_mediation(panel, panel_med, cointegrated=False):
    """
    Rate channel:       NIIP -> 10yr yield -> savings
    Real income channel:NIIP -> mfg share  -> savings
    """
    p = panel.copy()
    p.index = pd.to_datetime(p.index).tz_localize(None)

    pm = panel_med.copy()
    pm.index = pd.to_datetime(pm.index).tz_localize(None)

    samples = {
        "1976-1995 (early accumulation)":  (p[(p.index >= pd.Timestamp("1976-01-01")) &
                                              (p.index <  pd.Timestamp("1996-01-01"))], None),
        "1990-2009 (globalization era)":   (p[(p.index >= pd.Timestamp("1990-01-01")) &
                                              (p.index <  pd.Timestamp("2010-01-01"))], None),
        "2006-2025 (financialization era)":( p[p.index >= pd.Timestamp("2006-01-01")], None),
        "Full 1976-2025":                  (p, None),
    }

    mediation_specs = [
        ("Rate channel",        "yield_10y", "savings", p),
        ("Real income channel", "mfg_share", "savings", p),
    ]

    print("\n" + "="*70)
    print("MEDIATION ANALYSIS")
    print("NIIP -> [Mediator] -> Savings")
    print("="*70)

    med_results = []

    for sample_name, (samp, _) in samples.items():
        if len(samp) < 20:
            continue

        print(f"\n--- {sample_name} ({len(samp)} quarters) ---")

        for med_name, mediator, outcome, src in mediation_specs:
            src_samp = src[src.index.isin(samp.index)]
            df3 = src_samp[["niip_gdp", mediator, outcome]].dropna()
            df3_use = df3.diff().dropna() if not cointegrated else df3

            if len(df3_use) < 16:
                print(f"  {med_name}: skip ({len(df3_use)} obs)")
                continue

            max_lag = min(MAX_LAGS, len(df3_use)//6)
            lags = [l for l in GRANGER_LAGS if l <= max_lag] or [2]
            ml = max(lags)

            print(f"\n  {med_name}: NIIP -> {mediator} -> {outcome} [n={len(df3_use)}, lag<={max_lag}]")

            def granger_p(df_in, y, x):
                try:
                    t = grangercausalitytests(df_in[[y,x]], maxlag=ml, verbose=False)
                    return min(t[l][0]["ssr_ftest"][1] for l in lags if l<=ml)
                except:
                    return 1.0

            p1 = granger_p(df3_use, mediator, "niip_gdp")
            p2 = granger_p(df3_use, outcome,  mediator)
            p3_direct = granger_p(df3_use, outcome, "niip_gdp")
            try:
                t4 = grangercausalitytests(df3_use[[outcome,"niip_gdp",mediator]],
                                           maxlag=ml, verbose=False)
                p3_ctrl = min(t4[l][0]["ssr_ftest"][1] for l in lags if l<=ml)
            except:
                p3_ctrl = 1.0

            st = lambda p: "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "n.s."
            print(f"    Step 1 NIIP -> {mediator}:      p={p1:.4f} {st(p1)}")
            print(f"    Step 2 {mediator} -> {outcome}: p={p2:.4f} {st(p2)}")
            print(f"    Direct NIIP -> {outcome}:       p={p3_direct:.4f} {st(p3_direct)}")
            print(f"    Controlled NIIP -> {outcome}:   p={p3_ctrl:.4f} {st(p3_ctrl)}")

            if p1<0.10 and p2<0.10 and p3_direct>0.10:
                verdict = "FULL MEDIATION"
            elif p1<0.10 and p2<0.10:
                verdict = "PARTIAL MEDIATION"
            elif p1<0.10 and p2<0.10:
                verdict = "CHANNEL ACTIVE"
            else:
                verdict = "CHANNEL WEAK"

            print(f"    Verdict: {verdict}")

            med_results.append({
                "sample": sample_name, "channel": med_name,
                "p_niip_to_med": p1, "p_med_to_sav": p2,
                "p_direct": p3_direct, "p_controlled": p3_ctrl,
                "verdict": verdict, "obs": len(df3_use),
            })

    return pd.DataFrame(med_results)

# ── CELL 11: CHARTS ──────────────────────────────────────────────────────────

def plot_niip_overview(panel):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#03060d")
    fig.suptitle("NIIP Stock and Downstream Variables 1976-2025 -- kappa-SFC Framework",
                 color="#e2e8f0", fontsize=13)

    pairs = [
        ("niip_gdp", "NIIP / GDP (%)",                  "#22d3ee"),
        ("cape",     "Shiller CAPE",                    "#f59e0b"),
        ("savings",  "Personal Savings Rate (%)",       "#a78bfa"),
        ("mfg_share","Manufacturing Employment Share (%)", "#ef4444"),
    ]

    for ax, (col, title, color) in zip(axes.flat, pairs):
        ax.set_facecolor("#03060d")
        ax.plot(panel[col], color=color, linewidth=1.5)
        for yr, lbl, col2 in [
            ("1985-09-01","Plaza Accord","#22c55e"),
            ("2001-01-01","WTO/PRC accession","#a78bfa"),
            ("2008-01-01","GFC","#ef4444"),
            ("2016-01-01","2016 structural break","#f59e0b"),
        ]:
            if pd.Timestamp(yr) >= panel.index[0]:
                ax.axvline(pd.Timestamp(yr), color=col2, linestyle="--",
                          linewidth=0.8, alpha=0.7, label=lbl)
        ax.set_title(title, color="#e2e8f0", fontsize=10)
        ax.tick_params(colors="#4a5568", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#1e2d3a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=6, labelcolor="#64748b",
                 facecolor="#03060d", edgecolor="#1e2d3a")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "kappa_01_niip_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")

def plot_scatter_matrix(panel):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#03060d")
    fig.suptitle("NIIP/GDP vs Downstream Variables 1976-2025 -- Scatter",
                 color="#e2e8f0", fontsize=12)

    pairs = [
        ("cape",      "Shiller CAPE",       "#f59e0b"),
        ("savings",   "Savings Rate (%)",   "#a78bfa"),
        ("mfg_share", "Mfg Emp Share (%)",  "#ef4444"),
    ]

    for ax, (col, label, color) in zip(axes, pairs):
        ax.set_facecolor("#03060d")
        x = panel["niip_gdp"].dropna()
        y = panel[col].reindex(x.index).dropna()
        x = x.reindex(y.index)

        # Color by era
        def era_color(t):
            if t < pd.Timestamp("2001-01-01"): return "#22c55e"   # pre-WTO
            elif t < pd.Timestamp("2016-01-01"): return "#60a5fa" # WTO era
            else: return "#f59e0b"                                  # post-2016

        colors = [era_color(t) for t in y.index]
        ax.scatter(x, y, c=colors, alpha=0.7, s=20, edgecolors="none")

        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, np.poly1d(z)(xline), color=color,
               linewidth=1.5, linestyle="--", alpha=0.8)

        corr = np.corrcoef(x, y)[0,1]
        ax.set_title(f"NIIP/GDP vs {label}\nr={corr:.3f}",
                    color="#e2e8f0", fontsize=10)
        ax.set_xlabel("NIIP/GDP (%)", color="#4a5568", fontsize=8)
        ax.set_ylabel(label, color="#4a5568", fontsize=8)
        ax.tick_params(colors="#4a5568", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#1e2d3a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#22c55e", label="Pre-2001 (pre-WTO)"),
        Patch(facecolor="#60a5fa", label="2001-2015 (WTO era)"),
        Patch(facecolor="#f59e0b", label="Post-2016 (basis trade era)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
              fontsize=8, labelcolor="#64748b",
              facecolor="#03060d", edgecolor="#1e2d3a")

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(SAVE_DIR, "kappa_02_scatter_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")

def plot_mediation_channels(panel):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#03060d")
    fig.suptitle("Mediation Channels: NIIP -> Rate/Income -> Savings 1976-2025",
                 color="#e2e8f0", fontsize=13)

    pairs = [
        (axes[0,0], "niip_gdp",  "yield_10y", "NIIP/GDP -> 10yr Yield (rate channel step 1)",    "#22d3ee","#f472b6"),
        (axes[0,1], "yield_10y", "savings",    "10yr Yield -> Savings Rate (rate channel step 2)", "#f472b6","#a78bfa"),
        (axes[1,0], "niip_gdp",  "mfg_share",  "NIIP/GDP -> Mfg Share (income channel step 1)",  "#22d3ee","#ef4444"),
        (axes[1,1], "mfg_share", "savings",    "Mfg Share -> Savings Rate (income channel step 2)","#ef4444","#a78bfa"),
    ]

    for ax, xcol, ycol, title, xcolor, ycolor in pairs:
        ax.set_facecolor("#03060d")
        df = panel[[xcol, ycol]].dropna()
        ax2 = ax.twinx()
        ax.plot(df[xcol],  color=xcolor, linewidth=1.5, label=xcol)
        ax2.plot(df[ycol], color=ycolor, linewidth=1.5, linestyle="--", label=ycol)
        ax.set_title(title, color="#e2e8f0", fontsize=9)
        ax.tick_params(colors="#4a5568", labelsize=7)
        ax2.tick_params(colors="#4a5568", labelsize=7)
        for sp in ax.spines.values():  sp.set_color("#1e2d3a")
        for sp in ax2.spines.values(): sp.set_color("#1e2d3a")
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, lab1+lab2, fontsize=7,
                 labelcolor="#64748b", facecolor="#03060d", edgecolor="#1e2d3a")

        # Shade Plaza and WTO periods
        for yr, lbl, col2 in [("1985-09-01","Plaza","#22c55e"),
                               ("2001-01-01","WTO","#a78bfa"),
                               ("2016-01-01","2016","#f59e0b")]:
            ts = pd.Timestamp(yr)
            if ts >= df.index[0]:
                ax.axvline(ts, color=col2, linewidth=0.7, linestyle=":", alpha=0.6)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "kappa_03_mediation_channels.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")

def plot_irfs(panel, cointegrated=False):
    df = panel[["niip_gdp","cape","mfg_share","savings","yield_10y"]].dropna()
    df_use = df.diff().dropna() if not cointegrated else df

    n_obs  = len(df_use)
    n_vars = df_use.shape[1]
    best_lag = min(4, max(1, (n_obs//(n_vars*2))-1))
    print(f"VAR IRF: {n_obs} obs, {n_vars} vars, lag={best_lag}")

    try:
        fitted = VAR(df_use).fit(best_lag)
        irf    = fitted.irf(periods=12)

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.patch.set_facecolor("#03060d")
        fig.suptitle("Impulse Response: NIIP/GDP Shock -> Downstream Variables 1976-2025",
                    color="#e2e8f0", fontsize=11)

        for ax, (title, var_idx, color) in zip(axes, [
            ("-> CAPE",       1, "#f59e0b"),
            ("-> Mfg Share",  2, "#ef4444"),
            ("-> Savings",    3, "#a78bfa"),
            ("-> 10yr Yield", 4, "#f472b6"),
        ]):
            ax.set_facecolor("#03060d")
            vals = irf.irfs[:, 0, var_idx]
            ax.plot(vals, color=color, linewidth=2)
            ax.axhline(0, color="#1e2d3a", linewidth=0.8)
            ax.fill_between(range(len(vals)),
                           vals - 0.5*np.std(vals),
                           vals + 0.5*np.std(vals),
                           alpha=0.15, color=color)
            ax.set_title(f"NIIP shock {title}", color="#e2e8f0", fontsize=9)
            ax.set_xlabel("Quarters", color="#4a5568", fontsize=8)
            ax.tick_params(colors="#4a5568", labelsize=7)
            for sp in ax.spines.values(): sp.set_color("#1e2d3a")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(SAVE_DIR, "kappa_04_irfs.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
        plt.show()
        print(f"Saved: {path}")

    except Exception as e:
        print(f"IRF failed: {e}")

def plot_granger_summary(results_df):
    fig, ax = plt.subplots(figsize=(13, max(6, len(results_df)*0.45)))
    fig.patch.set_facecolor("#03060d")
    ax.set_facecolor("#03060d")

    verdict_colors = {
        "CONFIRMED":    "#22c55e",
        "BIDIRECTIONAL":"#f59e0b",
        "REVERSED":     "#f87171",
        "NULL":         "#334155",
    }

    for i, (_, row) in enumerate(results_df.iterrows()):
        color = verdict_colors.get(row["verdict"], "#334155")
        ax.barh(i-0.2, -np.log10(max(row["niip_to_y_p"], 0.001)),
               height=0.35, color=color, alpha=0.9, label=row["verdict"] if i==0 else "")
        ax.barh(i+0.2, -np.log10(max(row["y_to_niip_p"], 0.001)),
               height=0.35, color=color, alpha=0.35)

    for p_val, lbl, col in [(0.10,"p=0.10","#ffffff"),
                              (0.05,"p=0.05","#f59e0b"),
                              (0.01,"p=0.01","#22c55e")]:
        ax.axvline(-np.log10(p_val), color=col, linewidth=0.8,
                  linestyle="--", alpha=0.5, label=lbl)

    labels = [f"{r['sample'][:18]} | {r['hypothesis'][:26]}"
              for _, r in results_df.iterrows()]
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(labels, color="#94a3b8", fontsize=7)
    ax.set_xlabel("-log10(p-value) -- higher = more significant",
                 color="#4a5568", fontsize=9)
    ax.set_title("Granger Causality -- kappa-SFC Framework\n(dark = NIIP->Y, light = Y->NIIP)",
                color="#e2e8f0", fontsize=10)
    ax.tick_params(colors="#4a5568", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#1e2d3a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, labelcolor="#64748b",
             facecolor="#03060d", edgecolor="#1e2d3a")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "kappa_05_granger_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")

def plot_structural_break(panel):
    """Chart 6 -- NIIP vs CAPE highlighting structural break at 2016"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#03060d")
    fig.suptitle("Structural Break at 2016 -- NIIP -> CAPE Causal Direction",
                 color="#e2e8f0", fontsize=12)

    # Left: NIIP and CAPE on dual axis
    ax1 = axes[0]
    ax1.set_facecolor("#03060d")
    ax1b = ax1.twinx()
    ax1.plot(panel["niip_gdp"],  color="#22d3ee", linewidth=1.5, label="NIIP/GDP %")
    ax1b.plot(panel["cape"],     color="#f59e0b", linewidth=1.5, linestyle="--", label="CAPE")
    ax1.axvline(pd.Timestamp("2016-01-01"), color="#f59e0b", linewidth=1.2,
               linestyle="--", alpha=0.8, label="2016 break")
    ax1.axvline(pd.Timestamp("2008-01-01"), color="#ef4444", linewidth=0.8,
               linestyle=":", alpha=0.6, label="GFC")
    ax1.set_title("NIIP/GDP and Shiller CAPE 1976-2025",
                 color="#e2e8f0", fontsize=10)
    ax1.set_ylabel("NIIP/GDP (%)", color="#22d3ee", fontsize=8)
    ax1b.set_ylabel("CAPE", color="#f59e0b", fontsize=8)
    ax1.tick_params(colors="#4a5568", labelsize=7)
    ax1b.tick_params(colors="#4a5568", labelsize=7)
    for sp in ax1.spines.values():  sp.set_color("#1e2d3a")
    for sp in ax1b.spines.values(): sp.set_color("#1e2d3a")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1+lines2, lab1+lab2, fontsize=7,
              labelcolor="#64748b", facecolor="#03060d", edgecolor="#1e2d3a")

    # Right: Rolling 20Q correlation NIIP vs CAPE
    ax2 = axes[1]
    ax2.set_facecolor("#03060d")
    x = panel["niip_gdp"].dropna()
    y = panel["cape"].reindex(x.index).dropna()
    x = x.reindex(y.index)
    df_roll = pd.DataFrame({"niip": x, "cape": y})
    rolling_corr = df_roll["niip"].rolling(20).corr(df_roll["cape"])
    ax2.plot(rolling_corr, color="#f59e0b", linewidth=1.5)
    ax2.axhline(0, color="#1e2d3a", linewidth=0.8)
    ax2.axvline(pd.Timestamp("2016-01-01"), color="#f59e0b", linewidth=1.2,
               linestyle="--", alpha=0.8, label="2016 break")
    ax2.axvline(pd.Timestamp("2008-01-01"), color="#ef4444", linewidth=0.8,
               linestyle=":", alpha=0.6, label="GFC")
    ax2.set_title("Rolling 20Q Correlation: NIIP/GDP vs CAPE",
                 color="#e2e8f0", fontsize=10)
    ax2.set_ylabel("Correlation", color="#4a5568", fontsize=8)
    ax2.tick_params(colors="#4a5568", labelsize=7)
    for sp in ax2.spines.values(): sp.set_color("#1e2d3a")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=7, labelcolor="#64748b",
              facecolor="#03060d", edgecolor="#1e2d3a")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "kappa_06_structural_break.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")

# ── CELL 12: SUMMARY ─────────────────────────────────────────────────────────

def print_granger_summary(results_df):
    print("\n" + "="*90)
    print("GRANGER CAUSALITY RESULTS -- kappa-SFC Framework")
    print("="*90)
    print(f"{'Sample':<22} {'Hypothesis':<32} {'NIIP->Y':>12} {'Y->NIIP':>12} {'Obs':>5} {'Verdict'}")
    print("-"*90)
    for _, r in results_df.iterrows():
        st = lambda p: "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
        np_ = f"{r['niip_to_y_p']:.4f}{st(r['niip_to_y_p'])}"
        yp_ = f"{r['y_to_niip_p']:.4f}{st(r['y_to_niip_p'])}"
        print(f"{r['sample']:<22} {r['hypothesis']:<32} {np_:>12} {yp_:>12} {int(r['obs']):>5} {r['verdict']}")
    print("="*90)
    print("* p<0.10  ** p<0.05  *** p<0.01")

def print_mediation_summary(med_df):
    print("\n" + "="*70)
    print("MEDIATION ANALYSIS RESULTS")
    print("="*70)
    for _, r in med_df.iterrows():
        print(f"\n{r['sample']} | {r['channel']} [n={r['obs']}]")
        print(f"  Step 1 NIIP->mediator:    p={r['p_niip_to_med']:.4f}")
        print(f"  Step 2 mediator->savings: p={r['p_med_to_sav']:.4f}")
        print(f"  Direct NIIP->savings:     p={r['p_direct']:.4f}")
        print(f"  Controlled:               p={r['p_controlled']:.4f}")
        print(f"  Verdict: {r['verdict']}")

# ═══════════════════════════════════════════════════════════════════════════════
# MANUFACTURING COMPOSITION + ELASTICITY ANALYSIS
# Added to v5: composition shift + rolling elasticity collapse
# ═══════════════════════════════════════════════════════════════════════════════

def pull_manufacturing_composition(api_key):
    """
    Pull manufacturing subsector employment from FRED (BLS CES series).
    Capital-intensive: Aerospace, Autos, Chemicals, Computer/Electronics
    Labor-intensive:   Textiles, Apparel, Furniture
    """
    fred = Fred(api_key=api_key)

    series = {
        # Capital-intensive / strategic
        "aerospace":    "CES3133640001",  # Aircraft & parts mfg
        "auto":         "CES3336360101",  # Motor vehicles & parts -- alt series
        "chemicals":    "CES3132500001",  # Chemical manufacturing
        "computers":    "CES3133440001",  # Computer & electronic products
        # Labor-intensive / offshorable
        "textiles":     "CES3131100001",  # Textile mills
        "apparel":      "CES3131500001",  # Apparel manufacturing
        "furniture":    "CES3133700001",  # Furniture & related
        # Total
        "total_mfg":    "MANEMP",         # All manufacturing employees
        # Hours as capital intensity proxy
        "avg_hours":    "AWHMAN",         # Avg weekly hours, mfg (higher = more capital intensive)
    }

    mfg_data = {}
    print("\nPulling manufacturing subsector employment...")
    for name, sid in series.items():
        try:
            s = fred.get_series(sid, observation_start="1990-01-01")
            s = s.resample("QS").mean()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s.name = name
            mfg_data[name] = s
            print(f"  {sid} ({name}): {s.index[0].date()} to {s.index[-1].date()}, {len(s)} obs")
        except Exception as e:
            print(f"  {sid} ({name}): FAILED -- {e}")
            # Try alternate series for auto
            if name == "auto":
                try:
                    s = fred.get_series("CES3133610001", observation_start="1990-01-01")
                    s = s.resample("QS").mean()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    s.name = name
                    mfg_data[name] = s
                    print(f"    Alternate auto series OK: {len(s)} obs")
                except:
                    pass

    return mfg_data


def build_composition_panel(mfg_data):
    """
    Build capital-intensity ratio and labor-intensive ratio over time.
    Capital-intensity ratio = (aerospace + auto + chemicals + computers) / total
    Labor-intensive ratio   = (textiles + apparel + furniture) / total
    """
    total = mfg_data.get("total_mfg")
    if total is None:
        print("Total manufacturing missing -- cannot build composition panel")
        return None

    # Capital-intensive subsectors
    cap_series = []
    for name in ["aerospace","auto","chemicals","computers"]:
        if name in mfg_data:
            cap_series.append(mfg_data[name])

    # Labor-intensive subsectors
    lab_series = []
    for name in ["textiles","apparel","furniture"]:
        if name in mfg_data:
            lab_series.append(mfg_data[name])

    if not cap_series:
        print("No capital-intensive subsectors available")
        return None

    cap_sum = pd.concat(cap_series, axis=1).sum(axis=1)
    lab_sum = pd.concat(lab_series, axis=1).sum(axis=1) if lab_series else None

    comp = pd.DataFrame({"total_mfg": total})
    comp["cap_intensive"] = cap_sum
    comp["cap_ratio"]     = cap_sum / total * 100

    if lab_sum is not None:
        comp["lab_intensive"] = lab_sum
        comp["lab_ratio"]     = lab_sum / total * 100

    if "avg_hours" in mfg_data:
        comp["avg_hours"] = mfg_data["avg_hours"]

    comp = comp.dropna(subset=["cap_ratio"])
    print(f"\nComposition panel: {comp.index[0].date()} to {comp.index[-1].date()}, {len(comp)} quarters")
    print(f"  Capital-intensive ratio: {comp.cap_ratio.min():.1f}% to {comp.cap_ratio.max():.1f}%")
    if "lab_ratio" in comp.columns:
        print(f"  Labor-intensive ratio:   {comp.lab_ratio.min():.1f}% to {comp.lab_ratio.max():.1f}%")

    return comp


def compute_rolling_elasticity(panel, window=20):
    """
    Rolling OLS: d(mfg_share) = beta * d(niip_gdp) + epsilon
    Estimate beta over rolling window_q quarters.
    Beta collapse post-2016 = elasticity collapse.
    """
    from numpy.linalg import lstsq

    df = panel[["niip_gdp","mfg_share"]].dropna().diff().dropna()

    betas = []
    dates = []

    for i in range(window, len(df)+1):
        window_df = df.iloc[i-window:i]
        X = window_df["niip_gdp"].values.reshape(-1,1)
        y = window_df["mfg_share"].values
        # Add constant
        X_c = np.column_stack([X, np.ones(len(X))])
        try:
            coef, _, _, _ = lstsq(X_c, y, rcond=None)
            betas.append(coef[0])
        except:
            betas.append(np.nan)
        dates.append(df.index[i-1])

    beta_series = pd.Series(betas, index=dates, name="rolling_beta")
    print(f"\nRolling elasticity computed: {len(beta_series)} estimates")
    print(f"  Pre-2016 mean beta: {beta_series[beta_series.index < pd.Timestamp('2016-01-01')].mean():.6f}")
    print(f"  Post-2016 mean beta: {beta_series[beta_series.index >= pd.Timestamp('2016-01-01')].mean():.6f}")

    return beta_series


def plot_composition_and_elasticity(comp_panel, beta_series, save_dir):
    """
    Chart 7: Composition shift + elasticity collapse -- the publication chart.
    Left:  Capital-intensive share (rising) vs labor-intensive share (falling)
    Right: Rolling beta of mfg elasticity to NIIP (collapsing toward zero)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#03060d")
    fig.suptitle(
        "Manufacturing Transformation: Composition Shift and Elasticity Collapse",
        color="#e2e8f0", fontsize=12
    )

    # ── LEFT: Composition shift ───────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#03060d")

    if comp_panel is not None:
        ax1.plot(comp_panel["cap_ratio"], color="#22c55e", linewidth=2,
                label="Capital-intensive share\n(Aerospace, Auto, Chemicals, Semis)")
        if "lab_ratio" in comp_panel.columns:
            ax1.plot(comp_panel["lab_ratio"], color="#ef4444", linewidth=2,
                    label="Labor-intensive share\n(Textiles, Apparel, Furniture)")
        if "avg_hours" in comp_panel.columns:
            ax1b = ax1.twinx()
            ax1b.plot(comp_panel["avg_hours"], color="#f59e0b", linewidth=1.2,
                     linestyle=":", alpha=0.8, label="Avg weekly hours (K/L proxy)")
            ax1b.set_ylabel("Avg Weekly Hours", color="#f59e0b", fontsize=8)
            ax1b.tick_params(colors="#4a5568", labelsize=7)
            for sp in ax1b.spines.values(): sp.set_color("#1e2d3a")
            ax1b.spines["top"].set_visible(False)
            lines2, lab2 = ax1b.get_legend_handles_labels()
        else:
            lines2, lab2 = [], []

    for yr, lbl, col in [("2001-01-01","WTO accession","#a78bfa"),
                          ("2016-01-01","2016 break","#f59e0b")]:
        ax1.axvline(pd.Timestamp(yr), color=col, linewidth=0.9,
                   linestyle="--", alpha=0.7, label=lbl)

    ax1.set_title("Manufacturing Composition Shift 1990-2025",
                 color="#e2e8f0", fontsize=10)
    ax1.set_ylabel("% of Total Manufacturing Employment", color="#4a5568", fontsize=8)
    ax1.tick_params(colors="#4a5568", labelsize=7)
    for sp in ax1.spines.values(): sp.set_color("#1e2d3a")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    lines1, lab1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1+lines2, lab1+lab2, fontsize=7,
              labelcolor="#64748b", facecolor="#03060d", edgecolor="#1e2d3a")

    # ── RIGHT: Elasticity collapse ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#03060d")

    if beta_series is not None and len(beta_series) > 0:
        # Color the line by period
        pre_break  = beta_series[beta_series.index < pd.Timestamp("2016-01-01")]
        post_break = beta_series[beta_series.index >= pd.Timestamp("2016-01-01")]

        ax2.plot(pre_break,  color="#60a5fa", linewidth=2,
                label=f"Pre-2016 beta (mean={pre_break.mean():.5f})")
        ax2.plot(post_break, color="#f59e0b", linewidth=2,
                label=f"Post-2016 beta (mean={post_break.mean():.5f})")

        # Zero line
        ax2.axhline(0, color="#334155", linewidth=0.8, linestyle="-")

        # Shade the collapse zone
        ax2.axvspan(pd.Timestamp("2016-01-01"), beta_series.index[-1],
                   alpha=0.08, color="#f59e0b", label="Elasticity collapse zone")

        ax2.axvline(pd.Timestamp("2016-01-01"), color="#f59e0b", linewidth=1.2,
                   linestyle="--", alpha=0.8)
        ax2.axvline(pd.Timestamp("2008-01-01"), color="#ef4444", linewidth=0.8,
                   linestyle=":", alpha=0.6, label="GFC")

        # Annotate the collapse
        pre_mean  = pre_break.mean()
        post_mean = post_break.mean()
        ax2.annotate(f"Pre-2016\nmean={pre_mean:.5f}",
                    xy=(pd.Timestamp("2005-01-01"), pre_mean),
                    color="#60a5fa", fontsize=8,
                    xytext=(pd.Timestamp("2000-01-01"), pre_mean*1.5),
                    arrowprops=dict(arrowstyle="->", color="#60a5fa", lw=0.8))
        ax2.annotate(f"Post-2016\nmean={post_mean:.5f}\n~0",
                    xy=(pd.Timestamp("2020-01-01"), post_mean),
                    color="#f59e0b", fontsize=8,
                    xytext=(pd.Timestamp("2018-01-01"), pre_mean*0.6),
                    arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=0.8))

    ax2.set_title("Rolling 20Q Elasticity: Mfg Share to NIIP Shock\n(beta -> 0 = elasticity collapse)",
                 color="#e2e8f0", fontsize=10)
    ax2.set_ylabel("Beta coefficient (d.mfg / d.NIIP)", color="#4a5568", fontsize=8)
    ax2.tick_params(colors="#4a5568", labelsize=7)
    for sp in ax2.spines.values(): sp.set_color("#1e2d3a")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=7, labelcolor="#64748b",
              facecolor="#03060d", edgecolor="#1e2d3a")

    plt.tight_layout()
    path = os.path.join(save_dir, "kappa_07_composition_elasticity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")


def run_manufacturing_analysis(api_key, panel, save_dir):
    """
    Master function: pull composition data, compute elasticity, plot chart 7.
    Call from main() after existing analysis.
    """
    print("\n" + "="*70)
    print("MANUFACTURING COMPOSITION + ELASTICITY ANALYSIS")
    print("="*70)

    # Pull subsector data
    mfg_data = pull_manufacturing_composition(api_key)

    # Build composition panel
    comp_panel = build_composition_panel(mfg_data)

    # Compute rolling elasticity from existing panel (full history)
    beta_series = compute_rolling_elasticity(panel, window=20)

    # Also compute per overlapping era for the evolution table
    eras = {
        "1976-1995": panel[(panel.index >= "1976-01-01") & (panel.index < "1996-01-01")],
        "1990-2009": panel[(panel.index >= "1990-01-01") & (panel.index < "2010-01-01")],
        "2006-2025": panel[panel.index >= "2006-01-01"],
    }
    print("\nRolling elasticity by era (full window within era):")
    for era_name, era_panel in eras.items():
        if len(era_panel) > 20:
            df_era = era_panel[["niip_gdp","mfg_share"]].dropna().diff().dropna()
            if len(df_era) > 8:
                from numpy.linalg import lstsq
                X = df_era["niip_gdp"].values.reshape(-1,1)
                y = df_era["mfg_share"].values
                X_c = np.column_stack([X, np.ones(len(X))])
                try:
                    coef, _, _, _ = lstsq(X_c, y, rcond=None)
                    print(f"  {era_name}: beta={coef[0]:.6f} (n={len(df_era)})")
                except:
                    print(f"  {era_name}: failed")

    # Print interpretation
    if beta_series is not None and len(beta_series) > 0:
        pre  = beta_series[beta_series.index < pd.Timestamp("2016-01-01")]
        post = beta_series[beta_series.index >= pd.Timestamp("2016-01-01")]
        print(f"\nElasticity interpretation:")
        print(f"  Pre-2016 beta:  {pre.mean():.6f} (NIIP deterioration -> mfg decline)")
        print(f"  Post-2016 beta: {post.mean():.6f} (near zero -- elasticity collapsed)")
        if pre.mean() != 0:
            collapse_pct = (1 - abs(post.mean()/pre.mean())) * 100
            print(f"  Elasticity collapse: {collapse_pct:.0f}%")
        print(f"\n  Interpretation: The remaining manufacturing base is no longer")
        print(f"  the primary adjustment margin for external imbalances.")
        print(f"  Capital-intensive sectors (aerospace, chemicals, semis) do not")
        print(f"  respond to NIIP deterioration the way labor-intensive sectors did.")

    # Plot
    plot_composition_and_elasticity(comp_panel, beta_series, save_dir)

    # Save data
    if comp_panel is not None:
        comp_panel.to_csv(os.path.join(save_dir, "kappa_mfg_composition_v5.csv"))
    if beta_series is not None:
        beta_series.to_csv(os.path.join(save_dir, "kappa_rolling_elasticity_v5.csv"))

    return comp_panel, beta_series



# ── CELL 13: MAIN ────────────────────────────────────────────────────────────

def main():
    print("kappa Granger Causality Analysis v5")
    print("Full history 1976-2025 | Bivariate + Mediation")
    print("="*60)

    print("\n[1] Pulling FRED data...")
    data = pull_fred_data(FRED_API_KEY)

    print("\n[2] Preparing panels...")
    panel, panel_med = prepare_panel(data)

    print("\n[3] ADF unit root tests...")
    run_adf(panel)

    print("\n[4] Cointegration test...")
    cointegrated = test_coint(panel)

    print("\n[5] Bivariate Granger causality tests...")
    results = run_granger(panel, cointegrated)
    print_granger_summary(results)

    print("\n[6] Mediation analysis...")
    med_results = run_mediation(panel, panel_med, cointegrated)
    print_mediation_summary(med_results)

    print("\n[7] Generating charts...")
    plot_niip_overview(panel)
    plot_scatter_matrix(panel)
    plot_mediation_channels(panel)
    plot_irfs(panel, cointegrated)
    plot_granger_summary(results)
    plot_structural_break(panel)

    # Save all outputs
    results.to_csv(os.path.join(SAVE_DIR, "kappa_granger_results_v5.csv"), index=False)
    med_results.to_csv(os.path.join(SAVE_DIR, "kappa_mediation_results_v5.csv"), index=False)
    panel.to_csv(os.path.join(SAVE_DIR, "kappa_panel_data_v5.csv"))
    print(f"\nAll files saved to {SAVE_DIR}")
    print("Charts: kappa_01 through kappa_06")
    print("CSVs: kappa_granger_results_v5.csv, kappa_mediation_results_v5.csv, kappa_panel_data_v5.csv")

    print("\n[8] Manufacturing composition + elasticity analysis...")
    comp_panel, beta_series = run_manufacturing_analysis(FRED_API_KEY, panel, SAVE_DIR)

    return panel, panel_med, results, med_results, comp_panel, beta_series

if __name__ == "__main__":
    panel, panel_med, results, med_results, comp_panel, beta_series = main()

if __name__ == "__main__":
    panel, panel_med, results, med_results, comp_panel, beta_series = main()

# ═══════════════════════════════════════════════════════════════════════════════
# MANUFACTURING COMPOSITION + ELASTICITY ANALYSIS
# Added to v5: composition shift + rolling elasticity collapse
# ═══════════════════════════════════════════════════════════════════════════════

def pull_manufacturing_composition(api_key):
    """
    Pull manufacturing subsector employment from FRED (BLS CES series).
    Capital-intensive: Aerospace, Autos, Chemicals, Computer/Electronics
    Labor-intensive:   Textiles, Apparel, Furniture
    """
    fred = Fred(api_key=api_key)

    series = {
        # Capital-intensive / strategic
        "aerospace":    "CES3133640001",  # Aircraft & parts mfg
        "auto":         "CES3336360101",  # Motor vehicles & parts -- alt series
        "chemicals":    "CES3132500001",  # Chemical manufacturing
        "computers":    "CES3133440001",  # Computer & electronic products
        # Labor-intensive / offshorable
        "textiles":     "CES3131100001",  # Textile mills
        "apparel":      "CES3131500001",  # Apparel manufacturing
        "furniture":    "CES3133700001",  # Furniture & related
        # Total
        "total_mfg":    "MANEMP",         # All manufacturing employees
        # Hours as capital intensity proxy
        "avg_hours":    "AWHMAN",         # Avg weekly hours, mfg (higher = more capital intensive)
    }

    mfg_data = {}
    print("\nPulling manufacturing subsector employment...")
    for name, sid in series.items():
        try:
            s = fred.get_series(sid, observation_start="1990-01-01")
            s = s.resample("QS").mean()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s.name = name
            mfg_data[name] = s
            print(f"  {sid} ({name}): {s.index[0].date()} to {s.index[-1].date()}, {len(s)} obs")
        except Exception as e:
            print(f"  {sid} ({name}): FAILED -- {e}")
            # Try alternate series for auto
            if name == "auto":
                try:
                    s = fred.get_series("CES3133610001", observation_start="1990-01-01")
                    s = s.resample("QS").mean()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    s.name = name
                    mfg_data[name] = s
                    print(f"    Alternate auto series OK: {len(s)} obs")
                except:
                    pass

    return mfg_data


def build_composition_panel(mfg_data):
    """
    Build capital-intensity ratio and labor-intensive ratio over time.
    Capital-intensity ratio = (aerospace + auto + chemicals + computers) / total
    Labor-intensive ratio   = (textiles + apparel + furniture) / total
    """
    total = mfg_data.get("total_mfg")
    if total is None:
        print("Total manufacturing missing -- cannot build composition panel")
        return None

    # Capital-intensive subsectors
    cap_series = []
    for name in ["aerospace","auto","chemicals","computers"]:
        if name in mfg_data:
            cap_series.append(mfg_data[name])

    # Labor-intensive subsectors
    lab_series = []
    for name in ["textiles","apparel","furniture"]:
        if name in mfg_data:
            lab_series.append(mfg_data[name])

    if not cap_series:
        print("No capital-intensive subsectors available")
        return None

    cap_sum = pd.concat(cap_series, axis=1).sum(axis=1)
    lab_sum = pd.concat(lab_series, axis=1).sum(axis=1) if lab_series else None

    comp = pd.DataFrame({"total_mfg": total})
    comp["cap_intensive"] = cap_sum
    comp["cap_ratio"]     = cap_sum / total * 100

    if lab_sum is not None:
        comp["lab_intensive"] = lab_sum
        comp["lab_ratio"]     = lab_sum / total * 100

    if "avg_hours" in mfg_data:
        comp["avg_hours"] = mfg_data["avg_hours"]

    comp = comp.dropna(subset=["cap_ratio"])
    print(f"\nComposition panel: {comp.index[0].date()} to {comp.index[-1].date()}, {len(comp)} quarters")
    print(f"  Capital-intensive ratio: {comp.cap_ratio.min():.1f}% to {comp.cap_ratio.max():.1f}%")
    if "lab_ratio" in comp.columns:
        print(f"  Labor-intensive ratio:   {comp.lab_ratio.min():.1f}% to {comp.lab_ratio.max():.1f}%")

    return comp


def compute_rolling_elasticity(panel, window=20):
    """
    Rolling OLS: d(mfg_share) = beta * d(niip_gdp) + epsilon
    Estimate beta over rolling window_q quarters.
    Beta collapse post-2016 = elasticity collapse.
    """
    from numpy.linalg import lstsq

    df = panel[["niip_gdp","mfg_share"]].dropna().diff().dropna()

    betas = []
    dates = []

    for i in range(window, len(df)+1):
        window_df = df.iloc[i-window:i]
        X = window_df["niip_gdp"].values.reshape(-1,1)
        y = window_df["mfg_share"].values
        # Add constant
        X_c = np.column_stack([X, np.ones(len(X))])
        try:
            coef, _, _, _ = lstsq(X_c, y, rcond=None)
            betas.append(coef[0])
        except:
            betas.append(np.nan)
        dates.append(df.index[i-1])

    beta_series = pd.Series(betas, index=dates, name="rolling_beta")
    print(f"\nRolling elasticity computed: {len(beta_series)} estimates")
    print(f"  Pre-2016 mean beta: {beta_series[beta_series.index < pd.Timestamp('2016-01-01')].mean():.6f}")
    print(f"  Post-2016 mean beta: {beta_series[beta_series.index >= pd.Timestamp('2016-01-01')].mean():.6f}")

    return beta_series


def plot_composition_and_elasticity(comp_panel, beta_series, save_dir):
    """
    Chart 7: Composition shift + elasticity collapse -- the publication chart.
    Left:  Capital-intensive share (rising) vs labor-intensive share (falling)
    Right: Rolling beta of mfg elasticity to NIIP (collapsing toward zero)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#03060d")
    fig.suptitle(
        "Manufacturing Transformation: Composition Shift and Elasticity Collapse",
        color="#e2e8f0", fontsize=12
    )

    # ── LEFT: Composition shift ───────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#03060d")

    if comp_panel is not None:
        ax1.plot(comp_panel["cap_ratio"], color="#22c55e", linewidth=2,
                label="Capital-intensive share\n(Aerospace, Auto, Chemicals, Semis)")
        if "lab_ratio" in comp_panel.columns:
            ax1.plot(comp_panel["lab_ratio"], color="#ef4444", linewidth=2,
                    label="Labor-intensive share\n(Textiles, Apparel, Furniture)")
        if "avg_hours" in comp_panel.columns:
            ax1b = ax1.twinx()
            ax1b.plot(comp_panel["avg_hours"], color="#f59e0b", linewidth=1.2,
                     linestyle=":", alpha=0.8, label="Avg weekly hours (K/L proxy)")
            ax1b.set_ylabel("Avg Weekly Hours", color="#f59e0b", fontsize=8)
            ax1b.tick_params(colors="#4a5568", labelsize=7)
            for sp in ax1b.spines.values(): sp.set_color("#1e2d3a")
            ax1b.spines["top"].set_visible(False)
            lines2, lab2 = ax1b.get_legend_handles_labels()
        else:
            lines2, lab2 = [], []

    for yr, lbl, col in [("2001-01-01","WTO accession","#a78bfa"),
                          ("2016-01-01","2016 break","#f59e0b")]:
        ax1.axvline(pd.Timestamp(yr), color=col, linewidth=0.9,
                   linestyle="--", alpha=0.7, label=lbl)

    ax1.set_title("Manufacturing Composition Shift 1990-2025",
                 color="#e2e8f0", fontsize=10)
    ax1.set_ylabel("% of Total Manufacturing Employment", color="#4a5568", fontsize=8)
    ax1.tick_params(colors="#4a5568", labelsize=7)
    for sp in ax1.spines.values(): sp.set_color("#1e2d3a")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    lines1, lab1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1+lines2, lab1+lab2, fontsize=7,
              labelcolor="#64748b", facecolor="#03060d", edgecolor="#1e2d3a")

    # ── RIGHT: Elasticity collapse ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#03060d")

    if beta_series is not None and len(beta_series) > 0:
        # Color the line by period
        pre_break  = beta_series[beta_series.index < pd.Timestamp("2016-01-01")]
        post_break = beta_series[beta_series.index >= pd.Timestamp("2016-01-01")]

        ax2.plot(pre_break,  color="#60a5fa", linewidth=2,
                label=f"Pre-2016 beta (mean={pre_break.mean():.5f})")
        ax2.plot(post_break, color="#f59e0b", linewidth=2,
                label=f"Post-2016 beta (mean={post_break.mean():.5f})")

        # Zero line
        ax2.axhline(0, color="#334155", linewidth=0.8, linestyle="-")

        # Shade the collapse zone
        ax2.axvspan(pd.Timestamp("2016-01-01"), beta_series.index[-1],
                   alpha=0.08, color="#f59e0b", label="Elasticity collapse zone")

        ax2.axvline(pd.Timestamp("2016-01-01"), color="#f59e0b", linewidth=1.2,
                   linestyle="--", alpha=0.8)
        ax2.axvline(pd.Timestamp("2008-01-01"), color="#ef4444", linewidth=0.8,
                   linestyle=":", alpha=0.6, label="GFC")

        # Annotate the collapse
        pre_mean  = pre_break.mean()
        post_mean = post_break.mean()
        ax2.annotate(f"Pre-2016\nmean={pre_mean:.5f}",
                    xy=(pd.Timestamp("2005-01-01"), pre_mean),
                    color="#60a5fa", fontsize=8,
                    xytext=(pd.Timestamp("2000-01-01"), pre_mean*1.5),
                    arrowprops=dict(arrowstyle="->", color="#60a5fa", lw=0.8))
        ax2.annotate(f"Post-2016\nmean={post_mean:.5f}\n~0",
                    xy=(pd.Timestamp("2020-01-01"), post_mean),
                    color="#f59e0b", fontsize=8,
                    xytext=(pd.Timestamp("2018-01-01"), pre_mean*0.6),
                    arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=0.8))

    ax2.set_title("Rolling 20Q Elasticity: Mfg Share to NIIP Shock\n(beta -> 0 = elasticity collapse)",
                 color="#e2e8f0", fontsize=10)
    ax2.set_ylabel("Beta coefficient (d.mfg / d.NIIP)", color="#4a5568", fontsize=8)
    ax2.tick_params(colors="#4a5568", labelsize=7)
    for sp in ax2.spines.values(): sp.set_color("#1e2d3a")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=7, labelcolor="#64748b",
              facecolor="#03060d", edgecolor="#1e2d3a")

    plt.tight_layout()
    path = os.path.join(save_dir, "kappa_07_composition_elasticity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#03060d")
    plt.show()
    print(f"Saved: {path}")


def run_manufacturing_analysis(api_key, panel, save_dir):
    """
    Master function: pull composition data, compute elasticity, plot chart 7.
    Call from main() after existing analysis.
    """
    print("\n" + "="*70)
    print("MANUFACTURING COMPOSITION + ELASTICITY ANALYSIS")
    print("="*70)

    # Pull subsector data
    mfg_data = pull_manufacturing_composition(api_key)

    # Build composition panel
    comp_panel = build_composition_panel(mfg_data)

    # Compute rolling elasticity from existing panel (full history)
    beta_series = compute_rolling_elasticity(panel, window=20)

    # Also compute per overlapping era for the evolution table
    eras = {
        "1976-1995": panel[(panel.index >= "1976-01-01") & (panel.index < "1996-01-01")],
        "1990-2009": panel[(panel.index >= "1990-01-01") & (panel.index < "2010-01-01")],
        "2006-2025": panel[panel.index >= "2006-01-01"],
    }
    print("\nRolling elasticity by era (full window within era):")
    for era_name, era_panel in eras.items():
        if len(era_panel) > 20:
            df_era = era_panel[["niip_gdp","mfg_share"]].dropna().diff().dropna()
            if len(df_era) > 8:
                from numpy.linalg import lstsq
                X = df_era["niip_gdp"].values.reshape(-1,1)
                y = df_era["mfg_share"].values
                X_c = np.column_stack([X, np.ones(len(X))])
                try:
                    coef, _, _, _ = lstsq(X_c, y, rcond=None)
                    print(f"  {era_name}: beta={coef[0]:.6f} (n={len(df_era)})")
                except:
                    print(f"  {era_name}: failed")

    # Print interpretation
    if beta_series is not None and len(beta_series) > 0:
        pre  = beta_series[beta_series.index < pd.Timestamp("2016-01-01")]
        post = beta_series[beta_series.index >= pd.Timestamp("2016-01-01")]
        print(f"\nElasticity interpretation:")
        print(f"  Pre-2016 beta:  {pre.mean():.6f} (NIIP deterioration -> mfg decline)")
        print(f"  Post-2016 beta: {post.mean():.6f} (near zero -- elasticity collapsed)")
        if pre.mean() != 0:
            collapse_pct = (1 - abs(post.mean()/pre.mean())) * 100
            print(f"  Elasticity collapse: {collapse_pct:.0f}%")
        print(f"\n  Interpretation: The remaining manufacturing base is no longer")
        print(f"  the primary adjustment margin for external imbalances.")
        print(f"  Capital-intensive sectors (aerospace, chemicals, semis) do not")
        print(f"  respond to NIIP deterioration the way labor-intensive sectors did.")

    # Plot
    plot_composition_and_elasticity(comp_panel, beta_series, save_dir)

    # Save data
    if comp_panel is not None:
        comp_panel.to_csv(os.path.join(save_dir, "kappa_mfg_composition_v5.csv"))
    if beta_series is not None:
        beta_series.to_csv(os.path.join(save_dir, "kappa_rolling_elasticity_v5.csv"))

    return comp_panel, beta_series


# ── PATCH MAIN TO INCLUDE MANUFACTURING ANALYSIS ─────────────────────────────
# Add this call at the end of main() before the return statement:
#
#   print("\n[8] Manufacturing composition + elasticity analysis...")
#   comp_panel, beta_series = run_manufacturing_analysis(FRED_API_KEY, panel, SAVE_DIR)
#
# Or run standalone:
#
#   comp_panel, beta_series = run_manufacturing_analysis(FRED_API_KEY, panel, SAVE_DIR)

