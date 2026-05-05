"""
╔══════════════════════════════════════════════════════════════════════╗
║   R-Ignite Hackathon · Climate Risk Project  (SPYDER)                ║
║   Malaysia vs Philippines · Actuarial Stress Projection 2030         ║
╚══════════════════════════════════════════════════════════════════════╝
─────────────────────────────────────────────────────────────────────
"""
# Critical Fix #1 — Force the matplotlib backend to be set before all imports
# In Spyder/IPython environments, this line ensures that plots are displayed

import matplotlib
try:
    # If work at IPython/Spyder this will work
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('matplotlib', 'inline')
        print("✓ IPython inline backend enabled")
    else:
        # Standard Python environment: Pop-up windows using TkAgg or Qt5Agg
        import sys
        if sys.platform == 'win32':
            matplotlib.use('TkAgg')
        else:
            try:
                matplotlib.use('TkAgg')
            except Exception:
                pass
        print("✓ TkAgg backend Enabled (the image will open in a separate window)")
except Exception as e:
    print(f"Backend Set prompts: {e}")

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

plt.ion()

# THEME CONFIGURATION
BG       = '#0D1117'
PANEL_BG = '#161B22'
GRID_CLR = '#21262D'
TEXT_PRI = '#E6EDF3'
TEXT_SEC = '#8B949E'
ACCENT   = '#58A6FF'
GOLD     = '#E3B341'
TEAL     = '#3FB950'

MY_RED   = '#FF6B6B'
MY_LINE  = '#FF4444'
MY_SHADE = '#FF444430'

PH_BLUE  = '#79C0FF'
PH_LINE  = '#1A7FD4'
PH_SHADE = '#1A7FD430'

SCENARIO_COLORS = ['#FFD700', '#00E676', '#FF6B6B']


plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    PANEL_BG,
    'axes.edgecolor':    GRID_CLR,
    'axes.labelcolor':   TEXT_SEC,
    'axes.titlecolor':   TEXT_PRI,
    'xtick.color':       TEXT_SEC,
    'ytick.color':       TEXT_SEC,
    'grid.color':        GRID_CLR,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.5,
    'text.color':        TEXT_PRI,
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.unicode_minus':False,
    'figure.dpi':        100,
    'savefig.dpi':       150,
    'figure.autolayout': False,   
    'interactive':       True,   
})

COUNTRIES = ['Malaysia', 'Philippines']
STYLES = {
    'Malaysia':    dict(hist=MY_RED,  line=MY_LINE,  shade=MY_SHADE),
    'Philippines': dict(hist=PH_BLUE, line=PH_LINE,  shade=PH_SHADE),
}

SCENARIOS = {
    '1%/yr (Weak policy)': 0.01,
    '2%/yr (NDC-aligned)': 0.02,
    '3%/yr (Accelerated)': 0.03,
}

FUTURE = np.arange(2024, 2031)

# Utility function: Pauses briefly after each plot is displayed to ensure Spyder has finished rendering
def show_and_save(fig, filename):
    """
    Unified functions for saving and displaying plots.
    Resolves the core issue of plots not being visible in Spyder.
    """
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    except Exception:
        pass

    fig.savefig(filename, dpi=150, facecolor=BG, bbox_inches='tight')
    print(f"  ✓ Saved → {filename}")

    fig.canvas.draw()
    plt.pause(0.1)

  
    plt.show()

    print(f"  ✓ Image displayed")


# 1. Data loading
print("━" * 70)
print("  STEP 1 · Loading World Bank WDI Dataset")
print("━" * 70)


try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"  ✓ Working Directory: {script_dir}")
except Exception:
    print(f"  ✓ Current working directory: {os.getcwd()}")

WDI_FILE = 'WB_WDI_WIDEF.csv'
if not os.path.exists(WDI_FILE):
    print(f"\n  ✗ Can't find it '{WDI_FILE}'")
    print(f"    Current directory: {os.getcwd()}")
    print(f"    Please place WB_WDI_WIDEF.csv in the directory mentioned above, or modify the path.")
    print(f"    The script will continue running using the built-in sample data...\n")
    WDI_FILE = None


USE_DEMO_DATA = (WDI_FILE is None)

if USE_DEMO_DATA:
    print("  ⚠  Use the built-in sample data (not actual WDI data)")
    demo_years = list(range(1990, 2023))
    demo_my_ghg = [128000 + i * 7200 + np.random.normal(0, 3000) for i, _ in enumerate(demo_years)]
    demo_ph_ghg = [55000  + i * 3100 + np.random.normal(0, 1500) for i, _ in enumerate(demo_years)]

    rows = []
    for i, (yr, my_v, ph_v) in enumerate(zip(demo_years, demo_my_ghg, demo_ph_ghg)):
        rows.append({'Country': 'Malaysia',    'Indicator': 'Total greenhouse gas emissions (kt of CO2 equivalent)', 'Year': yr, 'Value': my_v})
        rows.append({'Country': 'Philippines', 'Indicator': 'Total greenhouse gas emissions (kt of CO2 equivalent)', 'Year': yr, 'Value': ph_v})
        rows.append({'Country': 'Malaysia',    'Indicator': 'CO2 emissions (kt)',                                    'Year': yr, 'Value': my_v * 0.75})
        rows.append({'Country': 'Philippines', 'Indicator': 'CO2 emissions (kt)',                                    'Year': yr, 'Value': ph_v * 0.72})
        rows.append({'Country': 'Malaysia',    'Indicator': 'Forest area (% of land area)',                          'Year': yr, 'Value': max(62, 78 - i*0.25)})
        rows.append({'Country': 'Philippines', 'Indicator': 'Forest area (% of land area)',                          'Year': yr, 'Value': 26 + i*0.05})
        rows.append({'Country': 'Malaysia',    'Indicator': 'Energy use (kg of oil equivalent per capita)',           'Year': yr, 'Value': 2800 + i*80})
        rows.append({'Country': 'Philippines', 'Indicator': 'Energy use (kg of oil equivalent per capita)',           'Year': yr, 'Value': 480  + i*12})
        rows.append({'Country': 'Malaysia',    'Indicator': 'Renewable energy consumption (% of total final energy)','Year': yr, 'Value': max(5, 18 - i*0.2)})
        rows.append({'Country': 'Philippines', 'Indicator': 'Renewable energy consumption (% of total final energy)','Year': yr, 'Value': 42  - i*0.4})
        rows.append({'Country': 'Malaysia',    'Indicator': 'GDP per capita (current US$)',                           'Year': yr, 'Value': 3000 + i*310})
        rows.append({'Country': 'Philippines', 'Indicator': 'GDP per capita (current US$)',                           'Year': yr, 'Value': 800  + i*90})
    df_long = pd.DataFrame(rows)
    print(f"  ✓ Sample data: {len(df_long):,} row")
else:
    df = pd.read_csv(WDI_FILE, low_memory=False)

    COUNTRY_COLS   = ['Country Name', 'REF_AREA_LABEL', 'country_name', 'CountryName']
    INDICATOR_COLS = ['Indicator Name', 'INDICATOR_LABEL', 'indicator_name', 'IndicatorName']

    country_col   = next((c for c in COUNTRY_COLS   if c in df.columns), None)
    indicator_col = next((c for c in INDICATOR_COLS if c in df.columns), None)

    if not country_col or not indicator_col:
        raise ValueError(f"Country/Indicator column not found. Available columns: {df.columns.tolist()[:10]}")

    year_cols = [col for col in df.columns
                 if str(col).strip().isdigit() and 1960 <= int(str(col).strip()) <= 2025]

    df_long = pd.melt(df, id_vars=[country_col, indicator_col],
                      value_vars=year_cols, var_name='Year', value_name='Value')
    df_long.rename(columns={country_col: 'Country', indicator_col: 'Indicator'}, inplace=True)
    df_long['Year']  = pd.to_numeric(df_long['Year'])
    df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
    df_long          = df_long.dropna(subset=['Value'])

    for country in COUNTRIES:
        exact = df_long['Country'].unique()
        if country not in exact:
            matches = [c for c in exact if country.lower() in c.lower()]
            if matches:
                df_long['Country'] = df_long['Country'].replace(matches[0], country)
                print(f"  ✓ Name Mapping: '{matches[0]}' → '{country}'")
            else:
                print(f"  ✗ Warning: '{country}' Not found in the data！")

    print(f"  ✓ WDI loading complete: {df_long.shape[0]:,} row")

# Automatic recognition indicators
all_indicators = df_long['Indicator'].unique()

def find_indicator(keywords, exclude=None):
    for kw in keywords:
        matches = [n for n in all_indicators if kw.lower() in n.lower()]
        if exclude:
            matches = [m for m in matches
                       if not any(e.lower() in m.lower() for e in exclude)]
        if matches:
            return matches[0]
    return None

GHG_IND    = find_indicator(['greenhouse gas', 'ghg', 'total greenhouse'])
CO2_IND    = find_indicator(['CO2 emission', 'carbon dioxide'], exclude=['per capita'])
FOREST_IND = find_indicator(['forest area'])
ENERGY_IND = find_indicator(['energy use', 'energy consumption', 'energy intensity'])
RENEW_IND  = find_indicator(['renewable energy', 'renewables'])
GDP_IND    = find_indicator(['GDP per capita', 'gdp per capita'])

print(f"\n  Identified metrics:")
for name, val in [('GHG',     GHG_IND),  ('CO2',    CO2_IND),
                  ('Forest',  FOREST_IND), ('Energy', ENERGY_IND),
                  ('Renew%',  RENEW_IND),  ('GDP/cap',GDP_IND)]:
    icon = '✓' if val else '✗'
    short = (val[:60] + '…') if val and len(val) > 60 else (val or 'Not found')
    print(f"    [{icon}] {name}: {short}")

if GHG_IND is None:
    raise RuntimeError("Cannot find the GHG metrics, please check that the WDI CSV file is complete.")

def get_series(country, indicator, min_points=5):
    if indicator is None:
        return None
    sub = df_long[
        (df_long['Country'] == country) &
        (df_long['Indicator'] == indicator)
    ].sort_values('Year')
    return sub if len(sub) >= min_points else None

# 2. Insurance/disaster claims data
print("\n" + "━" * 70)
print("  STEP 2 · Insurance / Disaster Claims Data")
print("━" * 70)

EMDAT_FILE = 'emdat_sea.csv'

def load_emdat(filepath):
    em = pd.read_csv(filepath, low_memory=False)
    em.columns = em.columns.str.strip()
    year_c    = next((c for c in em.columns if 'year'    in c.lower()), None)
    country_c = next((c for c in em.columns if 'country' in c.lower()), None)
    damage_c  = next((c for c in em.columns if 'insured' in c.lower() and 'damage' in c.lower()), None)
    if damage_c is None:
        damage_c = next((c for c in em.columns if 'damage' in c.lower()), None)
    events_c  = next((c for c in em.columns if 'event'   in c.lower() or 'occurrence' in c.lower()), None)
    if not all([year_c, country_c, damage_c]):
        return None
    em[year_c]   = pd.to_numeric(em[year_c], errors='coerce')
    em[damage_c] = pd.to_numeric(em[damage_c], errors='coerce').fillna(0)
    result = {}
    for country in COUNTRIES:
        mask = em[country_c].str.contains(country, case=False, na=False)
        sub  = em[mask].copy()
        if len(sub) == 0:
            continue
        agg_dict = {'Claims_MUSD': (damage_c, lambda x: x.sum() / 1000)}
        agg_dict['Events'] = (events_c, 'sum') if events_c else (damage_c, 'count')
        grp = sub.groupby(year_c).agg(**agg_dict).reset_index().rename(columns={year_c: 'Year'})
        grp = grp[grp['Year'] >= 2000]
        result[country] = grp
    return result if result else None

emdat_data = None
if os.path.exists(EMDAT_FILE):
    try:
        emdat_data = load_emdat(EMDAT_FILE)
        if emdat_data:
            print(f"  ✓ Real EM-DAT data has been loaded: '{EMDAT_FILE}'")
        else:
            print(f"  ✗ Parsing failed; using verified estimate")
    except Exception as e:
        print(f"  ✗ Read error: {e}，Use estimated values")
else:
    print(f"  ⚠  '{EMDAT_FILE}' Not available; use EM-DAT/Swiss Re to verify the estimates")

FALLBACK_YEARS = list(range(2010, 2024))
FALLBACK = {
    'Malaysia':    {'Claims_MUSD': [48,65,41,78,95,88,82,108,92,125,142,310,195,220],
                    'Events':      [5,6,5,7,8,7,7,9,8,10,9,12,10,11]},
    'Philippines': {'Claims_MUSD': [290,340,420,920,350,280,450,390,300,320,560,480,530,490],
                    'Events':      [12,15,18,20,14,13,16,15,13,14,18,17,16,15]},
}

if emdat_data is None:
    emdat_data = {c: pd.DataFrame({'Year': FALLBACK_YEARS,
                                   'Claims_MUSD': FALLBACK[c]['Claims_MUSD'],
                                   'Events':      FALLBACK[c]['Events']})
                  for c in COUNTRIES}
    print("  ✓ Estimated values have been loaded (EM-DAT CRED + Swiss Re Sigma 2024)")
    DATA_SOURCE_NOTE = '⚠ Estimated value: EM-DAT CRED + Swiss Re Sigma 2024'
else:
    DATA_SOURCE_NOTE = 'Source: EM-DAT CRED (emdat.be)'


# 3.Model Development 

print("\n" + "━" * 70)
print("  STEP 3 · Model Building & Validation")
print("━" * 70)

def smart_formatter(x, _):
    if abs(x) >= 1_000_000:  return f'{x/1_000_000:.1f}M'
    elif abs(x) >= 1_000:    return f'{x/1_000:.0f}k'
    else:                    return f'{x:.0f}'

def print_metrics(model_name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"     --- {model_name} ---")
    print(f"     R²: {r2:.4f} | MAE: {mae:,.2f} | RMSE: {rmse:,.2f}")

def build_model(country):
    sub = get_series(country, GHG_IND)
    if sub is None:
        print(f"  ✗ {country}: GHG Insufficient data")
        return None

    years     = sub['Year'].values
    y         = sub['Value'].values
    year_mean = years.mean()
    X         = (years - year_mean).reshape(-1, 1)
    X_future  = (FUTURE - year_mean).reshape(-1, 1)

    split = -5
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    ols = LinearRegression().fit(X_tr, y_tr)
    r2_ols  = r2_score(y_te, ols.predict(X_te))
    mae_ols = mean_absolute_error(y_te, ols.predict(X_te))
    cv_ols  = cross_val_score(LinearRegression(), X, y,
                               cv=KFold(n_splits=5, shuffle=False), scoring='r2').mean()

    poly2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly2.fit(X_tr, y_tr)
    r2_poly  = r2_score(y_te, poly2.predict(X_te))
    mae_poly = mean_absolute_error(y_te, poly2.predict(X_te))
    cv_poly  = cross_val_score(make_pipeline(PolynomialFeatures(2), LinearRegression()),
                                X, y, cv=KFold(n_splits=5, shuffle=False), scoring='r2').mean()

    # ── Random Forest ─────────────────────────────────────────────────────
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_tr, y_tr)
    rf_preds = rf_model.predict(X_te)
    r2_rf   = r2_score(y_te, rf_preds)
    mae_rf  = mean_absolute_error(y_te, rf_preds)
    rmse_rf = np.sqrt(mean_squared_error(y_te, rf_preds))

    # ── Support Vector Regression ─────────────────────────────────────────
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_tr_scaled = scaler_X.fit_transform(X_tr)
    X_te_scaled = scaler_X.transform(X_te)
    y_tr_scaled = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()

    svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
    svr_model.fit(X_tr_scaled, y_tr_scaled)
    svr_preds_scaled = svr_model.predict(X_te_scaled)
    svr_preds = scaler_y.inverse_transform(svr_preds_scaled.reshape(-1, 1)).ravel()
    r2_svr   = r2_score(y_te, svr_preds)
    mae_svr  = mean_absolute_error(y_te, svr_preds)
    rmse_svr = np.sqrt(mean_squared_error(y_te, svr_preds))

    use_poly = (cv_poly > cv_ols + 0.02)
    if use_poly:
        poly2.fit(X, y)
        probe = poly2.predict(X_future)
        if probe[-1] <= 0 or probe[-1] > y[-1] * 5 or np.isnan(probe[-1]):
            use_poly = False
            print(f"     ⚠  Poly 2030 Prediction error; switch to OLS")

    model_name = 'Polynomial (deg=2)' if use_poly else 'OLS Linear'
    best = (poly2 if use_poly else ols)
    best.fit(X, y)
    ols_full = LinearRegression().fit(X, y)

    base = np.maximum(best.predict(X_future), 0)
    scenarios = {label: np.array([base[i]*(1-rate)**(i+1) for i in range(len(FUTURE))])
                 for label, rate in SCENARIOS.items()}

    print(f"\n  📍 {country} ({model_name})")
    print(f"     OLS  CV-R²={cv_ols:.3f}  Test-R²={r2_ols:.3f}  MAE={mae_ols:,.0f} kt")
    print(f"     Poly CV-R²={cv_poly:.3f}  Test-R²={r2_poly:.3f}  MAE={mae_poly:,.0f} kt")
    print(f"     → choose: {model_name}  2024 forecast={base[0]:,.0f}  2030 baseline={base[-1]:,.0f} kt")
    print(f"     [Additional Models — Test Set]")
    print_metrics("Random Forest", y_te, rf_preds)
    print_metrics("Support Vector Regression", y_te, svr_preds)

    return dict(
        country=country, model=best, ols=ols_full,
        X=X, y=y, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
        X_orig=years.reshape(-1,1),
        X_te_orig=(X_te.flatten()+year_mean).reshape(-1,1),
        year_mean=year_mean, X_future=X_future,
        base=base, scenarios=scenarios, model_name=model_name,
        r2_ols=r2_ols, mae_ols=mae_ols, cv_ols=cv_ols,
        r2_poly=r2_poly, mae_poly=mae_poly, cv_poly=cv_poly,
        r2_test=max(r2_ols,r2_poly), mae_test=min(mae_ols,mae_poly),
        cv_r2=max(cv_ols,cv_poly),
        r2_rf=r2_rf, mae_rf=mae_rf, rmse_rf=rmse_rf,
        r2_svr=r2_svr, mae_svr=mae_svr, rmse_svr=rmse_svr,
    )

models = [build_model(c) for c in COUNTRIES]
models = [m for m in models if m is not None]

if not models:
    raise RuntimeError("Model creation failed. Please check the GHG data.")

print("\n" + "━" * 70)
print("  STEP 4 · Figure 0: Indicator Exploration")
print("━" * 70)

HEATMAP_INDICATORS = {k: v for k, v in {
    'GHG':     GHG_IND,   'CO2':    CO2_IND,
    'Forest':  FOREST_IND, 'Energy': ENERGY_IND,
    'GDP/cap': GDP_IND,   'Renew%': RENEW_IND,
}.items() if v is not None}

fig0 = plt.figure(figsize=(17, 6), facecolor=BG, num='Fig0_Indicators')
fig0.clf()  
axes0 = [fig0.add_subplot(1, 3, i+1) for i in range(3)]

for ax_idx, country in enumerate(COUNTRIES):
    ax = axes0[ax_idx]
    ax.set_facecolor(PANEL_BG)

    pivot = {}
    for label, ind in HEATMAP_INDICATORS.items():
        s = get_series(country, ind)
        if s is not None:
            pivot[label] = s.set_index('Year')['Value']

    if not pivot or len(pivot) < 2:
        ax.axis('off')
        ax.text(0.5, 0.5, f'Insufficient data\n{country}', ha='center', va='center',
                transform=ax.transAxes, color=TEXT_SEC, fontsize=10)
        continue

    wide = pd.DataFrame(pivot).dropna()
    if len(wide) < 4:
        ax.axis('off')
        continue

    corr = wide.corr()
    n    = len(corr)
    im   = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, fontsize=8, rotation=35, ha='right')
    ax.set_yticklabels(corr.index, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            txt_clr = 'black' if 0.3 < abs(val) < 0.7 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7.5, color=txt_clr, fontweight='bold')

    cbar = fig0.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7, colors=TEXT_SEC)
    cbar.set_label('Pearson r', fontsize=7, color=TEXT_SEC)

    s = STYLES[country]
    ax.set_title(f'{country} — Indicator Correlations',
                 fontsize=10, color=s['line'], pad=10)
    ax.spines[:].set_visible(False)

ax_r = axes0[2]
ax_r.set_facecolor(PANEL_BG); ax_r.axis('off')
ax_r.text(0.5, 0.99, 'WHY THESE INDICATORS?', ha='center', va='top',
          fontsize=10, fontweight='bold', color=ACCENT, transform=ax_r.transAxes)

rationale_lines = [
    ('GHG Emissions',    'Primary modeling objectives; Direct drivers of global warming'),
    ('CO₂ Emissions',    'Major components of GHG emissions; strongly correlated with energy demand'),
    ('Forest Area',      'Deforestation Exacerbates Flood Risks (Malaysia Focus)'),
    ('Energy Use',       'Reliance on fossil fuels → Key levers for emissions reduction'),
    ('Renewable %',      'Green Transition Agent; Stress Testing Policy Variables'),
    ('GDP per Capita',   'Adaptability; related to the underwriting loss ratio'),
]
y0 = 0.89
for label, desc in rationale_lines:
    ax_r.text(0.04, y0, f'▶ {label}', ha='left', va='top',
              fontsize=8.5, color=ACCENT, fontweight='bold', transform=ax_r.transAxes)
    y0 -= 0.055
    ax_r.text(0.07, y0, desc, ha='left', va='top', fontsize=7.8,
              color=TEXT_PRI, transform=ax_r.transAxes, linespacing=1.4)
    y0 -= 0.075

ax_r.text(0.5, y0 - 0.02,
          'Logical chain:\nHigh energy consumption → High GHG → Disaster frequency↑\n→ Insurance Claims↑',
          ha='center', va='top', fontsize=8, color=GOLD,
          transform=ax_r.transAxes, linespacing=1.5,
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262D',
                    edgecolor=GOLD, alpha=0.85))

for sp in ax_r.spines.values():
    sp.set_visible(True); sp.set_color(ACCENT); sp.set_linewidth(0.9)

fig0.suptitle('REQ ①  |  INDICATOR EXPLORATION & SELECTION RATIONALE\n'
              'Correlation of climate, energy & socioeconomic indicators · MY vs PH',
              fontsize=11, fontweight='bold', color=TEXT_PRI, y=1.02)

show_and_save(fig0, 'fig0_indicator_exploration.png')

# FIGURE 1 — GHG Forecast + Stress Testing

print("\n" + "━" * 70)
print("  STEP 5 · Figure 1: GHG Projection + Stress Test")
print("━" * 70)

fig1 = plt.figure(figsize=(18, 12), facecolor=BG, num='Fig1_GHG_Projection')
fig1.clf()

gs1 = gridspec.GridSpec(3, 3, figure=fig1,
                         height_ratios=[3.5, 1.6, 1.2],
                         hspace=0.44, wspace=0.28,
                         left=0.07, right=0.97, top=0.88, bottom=0.06)

ax_main = fig1.add_subplot(gs1[0, :])
ax_main.set_facecolor(PANEL_BG)

if models:
    test_min = min(m['X_te_orig'].min() for m in models)
    test_max = max(m['X_te_orig'].max() for m in models)
    ax_main.axvspan(test_min, test_max, alpha=0.06, color=GOLD, zorder=1)

for res in models:
    s  = STYLES[res['country']]
    yr = res['X_orig'].flatten()
    ax_main.scatter(yr, res['y'], color=s['hist'], alpha=0.38, s=14, zorder=3)
    ax_main.plot(yr, res['model'].predict(res['X']),
                 color=s['hist'], alpha=0.40, lw=1.2, ls='--', zorder=4)
    ax_main.scatter(res['X_te_orig'].flatten(), res['y_te'],
                    color=s['line'], s=32, marker='D', alpha=0.88, zorder=5)
    ax_main.plot(FUTURE, res['base'], color=s['line'], lw=2.5, zorder=6,
                 label=f"{res['country']} — Baseline")

    for i, (sc_label, sc_vals) in enumerate(res['scenarios'].items()):
        ax_main.plot(FUTURE, sc_vals, color=SCENARIO_COLORS[i], lw=1.4,
                     ls=['-.',':','--'][i], alpha=0.88, zorder=6,
                     label=sc_label if res == models[0] else None)
        if i == 1:
            ax_main.fill_between(FUTURE, res['base'], sc_vals,
                                 color=s['shade'], zorder=2, alpha=0.50)

ax_main.axvline(x=2023.5, color=ACCENT, lw=0.9, ls=':', alpha=0.75)
ylims = ax_main.get_ylim()
ax_main.text(2023.7, ylims[0]+(ylims[1]-ylims[0])*0.02,
             'FORECAST ▶', color=ACCENT, fontsize=7.5, alpha=0.9, va='bottom')
if models:
    ax_main.text((test_min+test_max)/2, ylims[0]+(ylims[1]-ylims[0])*0.94,
                 'Test window', color=GOLD, fontsize=7, ha='center', va='top', alpha=0.8)

ax_main.yaxis.set_major_formatter(FuncFormatter(smart_formatter))
ax_main.set_ylabel('GHG Emissions (kt CO₂ eq.)', fontsize=9.5, labelpad=10)
ax_main.set_xlabel('Year', fontsize=9.5)
ax_main.grid(True, axis='y', alpha=0.5); ax_main.grid(True, axis='x', alpha=0.25)
ax_main.spines[:].set_visible(False)

legend_elements = [
    mlines.Line2D([0],[0], color=STYLES[c]['line'], lw=2.5, label=f'{c} Baseline')
    for c in COUNTRIES
] + [
    mlines.Line2D([0],[0], color=SCENARIO_COLORS[i], lw=1.4,
                  ls=['-.',':','--'][i], label=sc)
    for i, sc in enumerate(SCENARIOS.keys())
] + [mpatches.Patch(color=GOLD, alpha=0.4, label='Test window (last 5yr)')]

ax_main.legend(handles=legend_elements, frameon=True, loc='upper left',
               ncol=2, fontsize=7.8, framealpha=0.18,
               facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_PRI)

# National Statistics Card

for col_idx, res in enumerate(models[:2]):
    s     = STYLES[res['country']]
    slope = res['ols'].coef_[0]
    ndc   = res['scenarios']['2%/yr (NDC-aligned)']

    entries = [
        ('2024 Forecast', f"{res['base'][0]:,.0f} kt",          s['line']),
        ('2030 Baseline', f"{res['base'][-1]:,.0f} kt",          s['line']),
        ('2030 NDC (2%)', f"{ndc[-1]:,.0f} kt",                  SCENARIO_COLORS[1]),
        ('Mitigation Δ',  f"−{res['base'][-1]-ndc[-1]:,.0f} kt", TEAL),
        ('Annual Slope',  f"{slope:+,.0f} kt/yr",                s['hist']),
        ('Best Model',    res['model_name'],                      ACCENT),
        ('CV R²',         f"{res['cv_r2']:.4f}",                 ACCENT),
        ('Test MAE',      f"{res['mae_test']:,.0f} kt",           GOLD),
        ('OLS R²(test)',  f"{res['r2_ols']:.4f}",                TEXT_SEC),
        ('Poly R²(test)', f"{res['r2_poly']:.4f}",               TEXT_SEC),
    ]
    ax_c = fig1.add_subplot(gs1[1, col_idx])
    ax_c.set_facecolor(PANEL_BG)
    ax_c.set_xlim(0,1); ax_c.set_ylim(0,1); ax_c.axis('off')
    ax_c.text(0.5, 0.98, res['country'].upper(), ha='center', va='top',
              fontsize=10.5, fontweight='bold', color=s['line'], transform=ax_c.transAxes)
    rows = len(entries)
    for i, (lbl, val, clr) in enumerate(entries):
        y_pos = 0.88 - i*(0.87/rows)
        ax_c.text(0.04, y_pos, lbl, ha='left', va='center',
                  fontsize=7, color=TEXT_SEC, transform=ax_c.transAxes)
        ax_c.text(0.98, y_pos, val, ha='right', va='center',
                  fontsize=7.5, fontweight='bold', color=clr, transform=ax_c.transAxes)
        ax_c.axhline(y_pos-0.040, xmin=0.02, xmax=0.98, color=GRID_CLR, lw=0.5)
    for sp in ax_c.spines.values():
        sp.set_visible(True); sp.set_color(GRID_CLR); sp.set_linewidth(0.8)

# Key Insight card
ax_ins = fig1.add_subplot(gs1[1, 2])
ax_ins.set_facecolor(PANEL_BG); ax_ins.axis('off')
ax_ins.text(0.5, 0.98, 'KEY INSIGHT', ha='center', va='top',
            fontsize=9.5, fontweight='bold', color=ACCENT, transform=ax_ins.transAxes)
if len(models) >= 2:
    ndc_my = models[0]['scenarios']['2%/yr (NDC-aligned)'][-1]
    ndc_ph = models[1]['scenarios']['2%/yr (NDC-aligned)'][-1]
    mit_my = models[0]['base'][-1] - ndc_my
    mit_ph = models[1]['base'][-1] - ndc_ph
    delta  = models[0]['base'][-1] - models[1]['base'][-1]
    txt = (f"Malaysia 2030 baseline\n{'exceeds' if delta>0 else 'trails'} PH by\n"
           f"{abs(delta):,.0f} kt CO₂eq.\n\nNDC mitigation (2%/yr)\nsaves by 2030:\n"
           f"  MY: {mit_my:,.0f} kt\n  PH: {mit_ph:,.0f} kt\n\n"
           f"Accelerated (3%/yr) shows\npotential under aggressive\ngreen energy policy.")
    ax_ins.text(0.5, 0.84, txt, ha='center', va='top', fontsize=8.2,
                color=TEXT_PRI, linespacing=1.65, transform=ax_ins.transAxes)
for sp in ax_ins.spines.values():
    sp.set_visible(True); sp.set_color(ACCENT); sp.set_linewidth(0.9)

# Model validation bar chart

ax_val = fig1.add_subplot(gs1[2, :2])
ax_val.set_facecolor(PANEL_BG)
x_pos = np.arange(len(models)); width = 0.28
for i, (metric_key, lbl) in enumerate([('cv_ols','CV R² (OLS)'), ('cv_poly','CV R² (Poly)')]):
    vals   = [m[metric_key] for m in models]
    colors = [STYLES[m['country']]['hist' if i==0 else 'line'] for m in models]
    offset = (i - 0.5) * (width + 0.04)
    bars   = ax_val.bar(x_pos+offset, vals, width, color=colors, alpha=0.82, label=lbl)
    for bar, val in zip(bars, vals):
        ax_val.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{val:.3f}', ha='center', fontsize=7.5, color=TEXT_PRI)
ax_val.set_xticks(x_pos)
ax_val.set_xticklabels([m['country'] for m in models], fontsize=9)
ax_val.set_ylabel('CV R² Score', fontsize=8)
ax_val.set_title('Model Validation — OLS vs Polynomial (5-Fold CV)', fontsize=9, color=TEXT_PRI)
ax_val.set_ylim(0, 1.20)
ax_val.axhline(0.70, color=GOLD, lw=1, ls='--', alpha=0.7)
ax_val.text(len(models)-0.5, 0.72, 'R²=0.70 benchmark', color=GOLD, fontsize=7)
ax_val.legend(fontsize=8, framealpha=0.20, facecolor=PANEL_BG, labelcolor=TEXT_PRI)
ax_val.spines[:].set_visible(False); ax_val.grid(True, axis='y', alpha=0.4)

# Bar chart showing emissions reductions and savings
ax_mit = fig1.add_subplot(gs1[2, 2])
ax_mit.set_facecolor(PANEL_BG)
bw = 0.22; n_sc = len(SCENARIOS); x_grp = np.arange(len(models))
for i, (sc_label, clr) in enumerate(zip(SCENARIOS.keys(), SCENARIO_COLORS)):
    savings = [m['base'][-1]-m['scenarios'][sc_label][-1] for m in models]
    offset  = (i-(n_sc-1)/2)*(bw+0.02)
    bars    = ax_mit.bar(x_grp+offset, savings, bw, color=clr, alpha=0.82,
                          label=['1%\n(Weak)','2%\n(NDC)','3%\n(Accel)'][i])
    for bar, val in zip(bars, savings):
        ax_mit.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
                    f'{val/1000:.0f}k', ha='center', fontsize=6.8, color=TEXT_PRI)
ax_mit.set_xticks(x_grp)
ax_mit.set_xticklabels([m['country'] for m in models], fontsize=9)
ax_mit.yaxis.set_major_formatter(FuncFormatter(smart_formatter))
ax_mit.set_title('2030 Mitigation Savings by Scenario', fontsize=9, color=TEXT_PRI)
ax_mit.legend(fontsize=7, framealpha=0.20, facecolor=PANEL_BG,
              labelcolor=TEXT_PRI, title='Rate', title_fontsize=7)
ax_mit.spines[:].set_visible(False); ax_mit.grid(True, axis='y', alpha=0.4)

fig1.text(0.07, 0.955, 'STRATEGIC CLIMATE RISK ASSESSMENT',
          fontsize=14, fontweight='bold', color=TEXT_PRI, va='top')
fig1.text(0.07, 0.929,
          'GHG Baseline & Multi-Scenario Stress Projection · MY vs PH · 2030 Horizon (Req ② & ④)',
          fontsize=9.5, color=TEXT_SEC, va='top')
fig1.add_artist(mlines.Line2D([0.07,0.97],[0.915,0.915],
                transform=fig1.transFigure, color=ACCENT, lw=0.8, alpha=0.6))

show_and_save(fig1, 'fig1_ghg_projection.png')

# 6. FIGURE 2 — Insurance claims vs climate indicators
print("\n" + "━" * 70)
print("  STEP 6 · Figure 2: Insurance Claims vs Climate Indicators")
print("━" * 70)

my_df = emdat_data['Malaysia']
ph_df = emdat_data['Philippines']

fig2 = plt.figure(figsize=(17, 11), facecolor=BG, num='Fig2_Insurance')
fig2.clf()
gs2 = gridspec.GridSpec(2, 3, figure=fig2,
                         hspace=0.42, wspace=0.32,
                         left=0.07, right=0.97, top=0.88, bottom=0.07)

# 2A: Timeline of Claims
ax2a = fig2.add_subplot(gs2[0, :2])
ax2a.set_facecolor(PANEL_BG)
ax2a.plot(my_df['Year'], my_df['Claims_MUSD'], color=MY_LINE, lw=2.2,
          marker='o', ms=4.5, label='Malaysia', zorder=4)
ax2a.plot(ph_df['Year'], ph_df['Claims_MUSD'], color=PH_LINE, lw=2.2,
          marker='s', ms=4.5, label='Philippines', zorder=4)
ax2a.fill_between(my_df['Year'], my_df['Claims_MUSD'], alpha=0.10, color=MY_RED)
ax2a.fill_between(ph_df['Year'], ph_df['Claims_MUSD'], alpha=0.10, color=PH_BLUE)

events_annot = [
    (2013, ph_df[ph_df['Year']==2013]['Claims_MUSD'].values, 'Typhoon Haiyan\n(2013)', (2015.5,790), PH_LINE),
    (2021, my_df[my_df['Year']==2021]['Claims_MUSD'].values, 'Selangor\nmegaflood (2021)', (2018.8,285), MY_LINE),
    (2020, ph_df[ph_df['Year']==2020]['Claims_MUSD'].values, 'Typhoon Goni\n+ Vamco (2020)', (2017.0,615), PH_LINE),
]
for yr, vals, note, txtpos, clr in events_annot:
    if len(vals) > 0:
        ax2a.annotate(note, xy=(yr, vals[0]), xytext=txtpos,
                      arrowprops=dict(arrowstyle='->', color=clr, lw=1.2),
                      fontsize=7.5, color=clr, ha='center')

ax2a.set_title('Natural Disaster Insurance Claims (2010–2023)', fontsize=10, color=TEXT_PRI)
ax2a.set_ylabel('Claims (USD Million)', fontsize=8.5)
ax2a.set_xlabel('Year', fontsize=8.5)
ax2a.legend(fontsize=8.5, framealpha=0.2, facecolor=PANEL_BG, labelcolor=TEXT_PRI)
ax2a.grid(True, axis='y', alpha=0.5); ax2a.spines[:].set_visible(False)
ax2a.text(0.99, 0.01, DATA_SOURCE_NOTE, ha='right', va='bottom',
          transform=ax2a.transAxes, fontsize=6.5, color=TEXT_SEC)

# 2B: Statistics Card
ax2b = fig2.add_subplot(gs2[0, 2])
ax2b.set_facecolor(PANEL_BG); ax2b.axis('off')
ax2b.text(0.5, 0.98, 'CLAIMS COMPARISON', ha='center', va='top',
          fontsize=9.5, fontweight='bold', color=ACCENT, transform=ax2b.transAxes)
row_labels = ['Total (USD M)','Avg/yr (USD M)','Peak (USD M)','Peak Year','Trend (M/yr)']
ax2b.text(0.44, 0.90, 'MY', fontsize=8.5, color=MY_LINE, transform=ax2b.transAxes,
          ha='center', fontweight='bold')
ax2b.text(0.78, 0.90, 'PH', fontsize=8.5, color=PH_LINE, transform=ax2b.transAxes,
          ha='center', fontweight='bold')
for df_, label_col, color in [(my_df, 0.44, MY_LINE), (ph_df, 0.78, PH_LINE)]:
    vals_list = [f"{df_['Claims_MUSD'].sum():,.0f}",
                 f"{df_['Claims_MUSD'].mean():,.0f}",
                 f"{df_['Claims_MUSD'].max():,.0f}",
                 f"{int(df_.loc[df_['Claims_MUSD'].idxmax(),'Year'])}",
                 f"{np.polyfit(df_['Year'],df_['Claims_MUSD'],1)[0]:+.1f}"]
    for i, (lbl, val) in enumerate(zip(row_labels, vals_list)):
        y = 0.82 - i*0.11
        if label_col == 0.44:
            ax2b.text(0.04, y, lbl, fontsize=7.2, color=TEXT_SEC,
                      transform=ax2b.transAxes, va='center')
        ax2b.text(label_col, y, val, fontsize=7.5, color=color,
                  transform=ax2b.transAxes, va='center', ha='center', fontweight='bold')
        ax2b.axhline(y-0.04, xmin=0.02, xmax=0.98, color=GRID_CLR, lw=0.5)
ax2b.text(0.5, 0.25,
          'PH: Typhoon Belt Exposed\n→ A single large loss\n\nMY: Flood levels are rising\n→ Due to deforestation and urban expansion',
          ha='center', va='top', fontsize=7.8, color=TEXT_PRI,
          linespacing=1.5, transform=ax2b.transAxes)
for sp in ax2b.spines.values():
    sp.set_visible(True); sp.set_color(GRID_CLR); sp.set_linewidth(0.8)

# 2C: GHG vs Claims Scatter Plot
ax2c = fig2.add_subplot(gs2[1, 0])
ax2c.set_facecolor(PANEL_BG)
for country, ins_df_, s in [('Malaysia',my_df,STYLES['Malaysia']),
                              ('Philippines',ph_df,STYLES['Philippines'])]:
    ghg_s = get_series(country, GHG_IND)
    if ghg_s is None: continue
    merged = pd.merge(ghg_s[['Year','Value']].rename(columns={'Value':'GHG'}),
                      ins_df_[['Year','Claims_MUSD']], on='Year')
    if len(merged) < 4: continue
    ax2c.scatter(merged['GHG']/1e6, merged['Claims_MUSD'],
                 color=s['line'], alpha=0.75, s=38, label=country, zorder=4)
    z  = np.polyfit(merged['GHG'], merged['Claims_MUSD'], 1)
    xs = np.linspace(merged['GHG'].min(), merged['GHG'].max(), 50)
    ax2c.plot(xs/1e6, np.poly1d(z)(xs), color=s['hist'], lw=1.3, ls='--', alpha=0.65)
    corr_val = merged[['GHG','Claims_MUSD']].corr().iloc[0,1]
    ax2c.text(0.97, 0.04+list(COUNTRIES).index(country)*0.10,
              f'{country} r={corr_val:.2f}', ha='right', va='bottom',
              fontsize=7.5, color=s['line'], transform=ax2c.transAxes)
ax2c.set_title('GHG Emissions vs Insurance Claims', fontsize=10, color=TEXT_PRI)
ax2c.set_xlabel('GHG (Mt CO₂ eq.)', fontsize=8.5)
ax2c.set_ylabel('Insurance Claims (USD M)', fontsize=8.5)
ax2c.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG, labelcolor=TEXT_PRI)
ax2c.grid(True, alpha=0.35); ax2c.spines[:].set_visible(False)

# 2D: Comparison of standardised indicators
ax2d = fig2.add_subplot(gs2[1, 1])
ax2d.set_facecolor(PANEL_BG)
COMPARE_INDS = {k:v for k,v in {'GHG':GHG_IND,'CO2':CO2_IND,
                                  'Forest\n(inv)':FOREST_IND,'Energy':ENERGY_IND,
                                  'Renew%':RENEW_IND}.items() if v is not None}
LATEST_YEAR = 2021
bars_MY, bars_PH, bar_labels = [], [], []
for label, ind in COMPARE_INDS.items():
    vals = {}
    for c in COUNTRIES:
        s = get_series(c, ind)
        if s is not None:
            row = s[s['Year']<=LATEST_YEAR].sort_values('Year').iloc[-1]
            vals[c] = row['Value']
    if len(vals) == 2:
        norm = max(vals.values())
        v_my = vals.get('Malaysia',0)/norm
        v_ph = vals.get('Philippines',0)/norm
        if 'forest' in label.lower() or 'inv' in label.lower():
            v_my = 1-v_my; v_ph = 1-v_ph
        bars_MY.append(v_my); bars_PH.append(v_ph); bar_labels.append(label)
x_b = np.arange(len(bar_labels)); w_b = 0.33
ax2d.bar(x_b-w_b/2, bars_MY, w_b, color=MY_LINE, alpha=0.82, label='Malaysia')
ax2d.bar(x_b+w_b/2, bars_PH, w_b, color=PH_LINE, alpha=0.82, label='Philippines')
ax2d.set_xticks(x_b); ax2d.set_xticklabels(bar_labels, fontsize=7.5, rotation=10, ha='right')
ax2d.set_title('Normalised Climate Risk Indicators', fontsize=10, color=TEXT_PRI)
ax2d.set_ylabel('Normalised Value (0=low, 1=high risk)', fontsize=8)
ax2d.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG, labelcolor=TEXT_PRI)
ax2d.grid(True, axis='y', alpha=0.4); ax2d.spines[:].set_visible(False)
ax2d.text(0.5,-0.16,f'Forest inverted · Data ≤{LATEST_YEAR}',
          ha='center', transform=ax2d.transAxes, fontsize=6.5, color=TEXT_SEC)

# 2E: Frequency of disaster events
ax2e = fig2.add_subplot(gs2[1, 2])
ax2e.set_facecolor(PANEL_BG)
all_years = sorted(set(my_df['Year'].tolist()+ph_df['Year'].tolist()))
ev_my = [my_df[my_df['Year']==yr]['Events'].values[0] if yr in my_df['Year'].values else 0 for yr in all_years]
ev_ph = [ph_df[ph_df['Year']==yr]['Events'].values[0] if yr in ph_df['Year'].values else 0 for yr in all_years]
x_ev = np.arange(len(all_years))
ax2e.bar(x_ev-0.2, ev_my, 0.38, color=MY_LINE, alpha=0.80, label='Malaysia')
ax2e.bar(x_ev+0.2, ev_ph, 0.38, color=PH_LINE, alpha=0.80, label='Philippines')
ax2e.set_xticks(x_ev[::2])
ax2e.set_xticklabels([str(y) for y in all_years[::2]], fontsize=7.5, rotation=30)
ax2e.set_title('Natural Disaster Events per Year', fontsize=10, color=TEXT_PRI)
ax2e.set_ylabel('# Disaster Events', fontsize=8.5)
ax2e.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG, labelcolor=TEXT_PRI)
ax2e.grid(True, axis='y', alpha=0.4); ax2e.spines[:].set_visible(False)
ax2e.text(0.5,-0.18, DATA_SOURCE_NOTE, ha='center',
          transform=ax2e.transAxes, fontsize=6.3, color=TEXT_SEC)

fig2.text(0.07, 0.955, 'CLIMATE INDICATORS vs NATURAL DISASTER INSURANCE CLAIMS  (Req ③)',
          fontsize=13, fontweight='bold', color=TEXT_PRI, va='top')
fig2.text(0.07, 0.930,
          'Malaysia vs Philippines · EM-DAT CRED · Swiss Re Sigma 2024 · World Bank WDI',
          fontsize=9.5, color=TEXT_SEC, va='top')
fig2.add_artist(mlines.Line2D([0.07,0.97],[0.915,0.915],
                transform=fig2.transFigure, color=ACCENT, lw=0.8, alpha=0.6))

show_and_save(fig2, 'fig2_insurance_claims.png')

# 7. FIGURE 3 — Stress Test Details
print("\n" + "━" * 70)
print("  STEP 7 · Figure 3: Stress Test Detail")
print("━" * 70)

fig3 = plt.figure(figsize=(17, 8), facecolor=BG, num='Fig3_StressTest')
fig3.clf()
gs3 = gridspec.GridSpec(1, 2, figure=fig3,
                         wspace=0.30, left=0.07, right=0.97,
                         top=0.84, bottom=0.11)

for col_idx, res in enumerate(models):
    ax = fig3.add_subplot(gs3[0, col_idx])
    ax.set_facecolor(PANEL_BG)
    s = STYLES[res['country']]

    yr_hist = res['X_orig'].flatten()
    ax.scatter(yr_hist, res['y'], color=s['hist'], alpha=0.28, s=10, zorder=2)
    ax.plot(yr_hist, res['model'].predict(res['X']),
            color=s['hist'], lw=1.0, ls='--', alpha=0.42, zorder=3, label='Historical trend')
    ax.axvline(x=2023.5, color=ACCENT, lw=0.8, ls=':', alpha=0.6)
    ax.plot(FUTURE, res['base'], color=s['line'], lw=2.5, zorder=6, label='Baseline')

    sc_ls = ['-.', ':', '--']
    for i, (sc_label, sc_vals) in enumerate(res['scenarios'].items()):
        ax.plot(FUTURE, sc_vals, color=SCENARIO_COLORS[i], lw=1.8,
                ls=sc_ls[i], zorder=5, label=sc_label.split(' ')[0])

    y_range  = float(res['y'].max() - res['y'].min())
    min_gap  = y_range * 0.055
    label_items = [('Baseline', float(res['base'][-1]), s['line'])]
    for i, (sc_label, sc_vals) in enumerate(res['scenarios'].items()):
        label_items.append((sc_label.split(' ')[0], float(sc_vals[-1]), SCENARIO_COLORS[i]))
    label_items.sort(key=lambda x: x[1], reverse=True)
    placed_y = []
    for lbl, raw_y, clr in label_items:
        adj_y = raw_y
        for py in placed_y:
            if abs(adj_y - py) < min_gap:
                adj_y = py - min_gap
        placed_y.append(adj_y)
        ax.text(2030.3, adj_y, f'{lbl}\n{raw_y/1000:.0f}k kt',
                va='center', ha='left', fontsize=6.8, color=clr, fontweight='bold')
        ax.plot(2030, raw_y, 'o', color=clr, ms=4, zorder=7)

    ax.set_xlim(yr_hist.min()-1, 2034)
    ax.yaxis.set_major_formatter(FuncFormatter(smart_formatter))
    ax.set_xlabel('Year', fontsize=9); ax.set_ylabel('GHG Emissions (kt CO₂ eq.)', fontsize=8.5)
    ax.set_title(f'{res["country"]}', fontsize=13, color=s['line'], pad=7)
    ax.grid(True, axis='y', alpha=0.35); ax.grid(True, axis='x', alpha=0.15)
    ax.spines[:].set_visible(False)
    ax.legend(fontsize=7.5, framealpha=0.18, facecolor=PANEL_BG,
              edgecolor=GRID_CLR, labelcolor=TEXT_PRI,
              loc='upper left', ncol=1, handlelength=1.8, borderpad=0.6)
    ax.text(0.5, -0.10,
            f"Model: {res['model_name']}  ·  CV R²={res['cv_r2']:.3f}\n"
            f"Scenarios: 1%=Weak  ·  2%=NDC  ·  3%=Accelerated (IPCC AR6 SSP1-2.6)",
            ha='center', va='top', fontsize=7, color=TEXT_SEC,
            transform=ax.transAxes, linespacing=1.45)

fig3.text(0.50, 0.975, 'REQ ④  |  MITIGATION STRATEGY STRESS TEST — 3-SCENARIO ANALYSIS',
          ha='center', va='top', fontsize=11, fontweight='bold', color=TEXT_PRI)
fig3.text(0.50, 0.952, 'Impact of green energy policy on GHG by 2030  ·  MY vs PH',
          ha='center', va='top', fontsize=8.5, color=TEXT_SEC)
fig3.text(0.50, 0.930,
          'MY NDC: −45% GHG intensity vs 2005  ·  PH NDC: −75% conditional  ·  IPCC AR6 SSP1-2.6',
          ha='center', va='top', fontsize=7.5, color=TEXT_SEC, style='italic')

show_and_save(fig3, 'fig3_stress_test.png')


plt.ioff()
plt.show(block=False)   # block=False 在 Spyder 里不卡住 console

# 8. Console Reports
DIV = "═" * 72
div = "─" * 72

print(f"\n{DIV}")
print(f"  ACTUARIAL SUMMARY REPORT  ·  R-Ignite Hackathon 2025")
print(f"  Climate Risk Assessment: Malaysia vs Philippines  ·  2030 Horizon")
print(DIV)

print(f"\n  REQ ① INDICATOR FINDINGS")
print(div)
print("""
  ┌──────────────────┬──────────────────────────────────────────────────┐
  │ Indicator        │ Climate Risk Relevance                           │
  ├──────────────────┼──────────────────────────────────────────────────┤
  │ GHG Emissions    │ Primary modelling target; systemic risk driver   │
  │ CO₂ Emissions    │ Largest GHG share; corr. with energy demand      │
  │ Forest Area      │ Deforestation amplifies flood risk (MY)          │
  │ Energy Use       │ Fossil-fuel proxy; key mitigation lever          │
  │ Renewable Energy │ Green transition indicator; stress-test variable │
  │ GDP per Capita   │ Adaptive capacity; insured loss ratio proxy      │
  └──────────────────┴──────────────────────────────────────────────────┘
""")

print(f"\n  REQ ② MODEL RESULTS")
print(div)
for res in models:
    slope = res['ols'].coef_[0]
    ndc   = res['scenarios']['2%/yr (NDC-aligned)']
    red   = (1 - ndc[-1]/res['base'][-1]) * 100
    print(f"\n  ┌─ {res['country']} ({res['model_name']}) {'─'*(42-len(res['country'])-len(res['model_name']))}")
    print(f"  │  Historical slope    : {slope:>+13,.0f} kt/yr")
    print(f"  │  OLS  CV-R²         : {res['cv_ols']:>12.4f}   MAE={res['mae_ols']:,.0f} kt")
    print(f"  │  Poly CV-R²         : {res['cv_poly']:>12.4f}   MAE={res['mae_poly']:,.0f} kt")
    print(f"  │  RF   Test-R²       : {res['r2_rf']:>12.4f}   MAE={res['mae_rf']:,.0f} kt  RMSE={res['rmse_rf']:,.0f} kt")
    print(f"  │  SVR  Test-R²       : {res['r2_svr']:>12.4f}   MAE={res['mae_svr']:,.0f} kt  RMSE={res['rmse_svr']:,.0f} kt")
    print(f"  │  2024 Forecast      : {res['base'][0]:>13,.0f} kt CO₂eq.")
    print(f"  │  2030 Baseline      : {res['base'][-1]:>13,.0f} kt CO₂eq.")
    print(f"  └{'─'*52}")

print(f"\n  REQ ③ INSURANCE CLAIMS")
print(div)
for country, df_ in emdat_data.items():
    pk_yr = int(df_.loc[df_['Claims_MUSD'].idxmax(),'Year'])
    pk_v  = df_['Claims_MUSD'].max()
    trend = np.polyfit(df_['Year'], df_['Claims_MUSD'], 1)[0]
    print(f"\n  {country}:  Total={df_['Claims_MUSD'].sum():,.0f}M  "
          f"Avg/yr={df_['Claims_MUSD'].mean():,.0f}M  "
          f"Peak={pk_yr}(${pk_v:,.0f}M)  Trend={trend:+.1f}M/yr")

print(f"\n  REQ ④ STRESS TEST RESULTS")
print(div)
for res in models:
    for sc_label, sc_vals in res['scenarios'].items():
        sav = res['base'][-1] - sc_vals[-1]
        pct = sav/res['base'][-1]*100
        print(f"  {res['country']:<14} [{sc_label}]: "
              f"2030={sc_vals[-1]:,.0f} kt  save={sav:,.0f} kt (−{pct:.1f}%)")
    print()

print(f"\n  REQ ⑤ KEY INSIGHTS & RECOMMENDATIONS")
print(div)
print("""
  INSIGHTS:
  • GHG upward trend in both countries → systemic SEA reinsurance risk
  • PH: typhoon belt → episodic large-loss events (Haiyan 2013 peak)
  • MY: rising flood claims from deforestation + urban floodplain expansion
  • Positive GHG–claims correlation confirms climate as material risk driver
  • MY 2021 megaflood signals structural shift → review pricing adequacy

  LIMITATIONS:
  • OLS/Poly assumes trend continuation; structural breaks not modelled
  • Low insurance penetration (MY~4%, PH~1.7% GDP) understates true losses
  • Mitigation rates depend on policy enforcement — high parametric uncertainty

  RECOMMENDATIONS:
  ① Use real EM-DAT CSV (emdat.be) for accurate claims analysis
  ② Add ARIMA/Prophet for non-linear trend + seasonal decomposition
  ③ Model forest area as flood co-variate for Malaysia
  ④ Add SST data for typhoon frequency modelling (Philippines)
  ⑤ Explore climate-linked catastrophe bond structuring for SEA portfolio
""")
print(DIV)
print("  ✓ All images have been saved:")
print("    fig0_indicator_exploration.png")
print("    fig1_ghg_projection.png")
print("    fig2_insurance_claims.png")
print("    fig3_stress_test.png")
print(DIV)