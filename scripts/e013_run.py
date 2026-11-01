# E013 — прогон по предрегистрации docs/E013_PREREGISTRATION.md (коммит e4668f1,
# заморожен ДО скачивания holdout). Однократный прогон, вердикт пересмотру не подлежит.
import numpy as np
import pandas as pd
from zipfile import ZipFile, is_zipfile

RAW = "/home/z/my-project/binfut-platform-restore/data/raw/klines1h"
HPY = 8760
RT = 4.5  # bps round-trip maker
PANEL_START = '2025-12-01'   # warmup; P&L с 2026-01-01
PNL_START = '2026-01-01'
CORE = ['BTCUSDT', 'ETHUSDT']
ALTS = ['SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
        'DOGEUSDT', 'LINKUSDT', 'LTCUSDT', 'DOTUSDT']
ALL10 = CORE + ALTS


def load_1h(sym, y0=2025, y1=2026):
    """Месячные архивы y0-12..2026-09 + дневные 2026-10 (28-е пропущен биржей)."""
    parts = []
    months = [(y0, 12)] + [(y1, m) for m in range(1, 10)]
    for y, m in months:
        p = f"{RAW}/{sym}-1h-{y}-{m:02d}.zip"
        if is_zipfile(p):
            parts.append(_read(p))
    for d in range(1, 32):
        p = f"{RAW}/{sym}-1h-2026-10-{d:02d}.zip"
        if is_zipfile(p):
            parts.append(_read(p))
    out = pd.concat(parts).sort_index()
    assert out.index.is_unique, sym
    return out['close'].astype(float)


def _read(p):
    with ZipFile(p) as z:
        with z.open(z.namelist()[0]) as f:
            head = f.readline().decode()
            f.seek(0)
            df = pd.read_csv(f, header=0 if head.lower().startswith('open_time') else None,
                             names=['open_time', 'open', 'high', 'low', 'close',
                                    'volume', 'ct', 'qvol', 'n', 'tbv', 'tbq', 'ig'],
                             usecols=[0, 4])
    return df.set_index(pd.to_datetime(df['open_time'], unit='ms'))


def stats(ser, name):
    ann = ser.mean() * HPY
    sh = ser.mean() / ser.std() * np.sqrt(HPY) if ser.std() > 0 else np.nan
    eq = (1 + ser).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = len(ser) / HPY
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    print(f"  {name:<26} net={ann:+8.1%}/год CAGR={cagr:+6.0%}  Sharpe={sh:+.2f}  maxDD={dd:.0%}")
    return ann, sh, dd, cagr


close = pd.DataFrame({s: load_1h(s) for s in ALL10}).loc[PANEL_START:]
r = np.log(close).diff()
vol = r.rolling(720).std() * np.sqrt(HPY)
step_ok = close.index.to_series().diff() <= pd.Timedelta(hours=1)
hold = slice(PNL_START, None)
print(f"панель: {close.index[0]} .. {close.index[-1]}, {len(close)} часов")

# ---------- §2 первичный рукав: XS R=168 ----------
R = 168
L = np.log(close / close.shift(720))
n = len(close)
rebal = np.zeros(n, bool)
rebal[::R] = True
W = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
Li, Ci = L.values, close.columns.tolist()
for i in np.where(rebal)[0]:
    if i < 720:
        continue
    row = pd.Series(Li[i], index=Ci).dropna()
    avail = [s for s in Ci if not (np.isnan(L[s].iloc[i]) or np.isnan(vol[s].iloc[i]))]
    if len(avail) < 6:
        continue
    ranks = row.loc[avail].rank(ascending=False)
    w = pd.Series(0.0, index=Ci)
    for s in ranks.head(3).index:
        w[s] = +min((0.30 / vol[s].iloc[i]) / 6, 1.0)
    for s in ranks.tail(3).index:
        w[s] = -min((0.30 / vol[s].iloc[i]) / 6, 1.0)
    W.iloc[i] = w.values
Wf = W.ffill().fillna(0.0)
turn = Wf.diff().abs().sum(axis=1).iloc[1:]
turn.iloc[0] = Wf.iloc[0].abs().sum()
xs = ((Wf.shift(1) * r.fillna(0.0)).sum(axis=1) - turn * RT / 2 / 1e4).where(step_ok).dropna()
xs_h = xs.loc[hold]
print("\n=== §2 Первичный рукав: XS-моментум R=168 (holdout 2026-01..10) ===")
xs_ann, xs_sh, xs_dd, _ = stats(xs_h, "XS R=168")
h1 = xs_h.loc['2026-01-01':'2026-06-30']
h2 = xs_h.loc['2026-07-01':]
print(f"  H1: {h1.sum()*1e2:+.1f}%  H2: {h2.sum()*1e2:+.1f}%  (порог: обе > 0)")
p_sh = (Wf.shift(1) * r.fillna(0.0)).sum(axis=1)
gross = Wf.abs().sum(axis=1).loc[hold].mean()
print(f"  ср. гросс: {gross:.2f}; месяцев: "
      + "/".join(f"{v:+.0%}" for v in ((1 + xs_h).groupby(xs_h.index.to_period('M')).prod() - 1).round(2).values))

# ---------- §3 вторичный рукав: альт-TSMOM K=168 ----------
K = 168
print("\n=== §3 Вторичный рукав: альт-TSMOM K=168 (holdout) ===")
streams, per_alt = [], {}
for s in ALTS:
    c = close[s].dropna()
    rr = np.log(c).diff()
    ok = c.index.to_series().diff() <= pd.Timedelta(hours=1)
    M = np.log(c / c.shift(K))
    P = (np.sign(M) * (0.30 / (rr.rolling(720).std() * np.sqrt(HPY)))).clip(upper=3.0).fillna(0.0)
    dpos = P.diff().abs().fillna(P.abs())
    st = (P.shift(1) * rr - dpos * RT / 2 / 1e4).where(ok).dropna().loc[hold]
    a, sh_, d_, _ = stats(st, s)
    per_alt[s] = a
    streams.append(st)
alt_b = pd.concat(streams, axis=1).mean(axis=1).dropna()
ab_ann, ab_sh, ab_dd, _ = stats(alt_b, "корзина 8 альтов")
pos_n = sum(1 for v in per_alt.values() if v > 0)

# ---------- §4 отчёты: комбо + сетка волтаргета ----------
print("\n=== §4a Комбо EW(core K720 + альты K168 + XS), holdout ===")
cs = []
for s in CORE:
    c = close[s].dropna()
    rr = np.log(c).diff()
    ok = c.index.to_series().diff() <= pd.Timedelta(hours=1)
    M = np.log(c / c.shift(720))
    P = (np.sign(M) * (0.30 / (rr.rolling(720).std() * np.sqrt(HPY)))).clip(upper=3.0).fillna(0.0)
    dpos = P.diff().abs().fillna(P.abs())
    cs.append((P.shift(1) * rr - dpos * RT / 2 / 1e4).where(ok).dropna().loc[hold])
core_h = pd.concat(cs, axis=1).mean(axis=1).dropna()
stats(core_h, "core BTC/ETH K720")
combo = pd.concat([core_h, alt_b, xs_h], axis=1).mean(axis=1).dropna()
_, cb_sh, cb_dd, cb_cagr = stats(combo, "КОМБО EW")

print("\n=== §4b Сетка волтаргета E012 core (полная история 2020-01..2026-10, IN-SAMPLE отчёт) ===")
def load_full(sym):
    parts = []
    for y in range(2020, 2027):
        for m in range(1, 13):
            p = f"{RAW}/{sym}-1h-{y}-{m:02d}.zip"
            if is_zipfile(p):
                parts.append(_read(p))
    for d in range(1, 32):
        p = f"{RAW}/{sym}-1h-2026-10-{d:02d}.zip"
        if is_zipfile(p):
            parts.append(_read(p))
    out = pd.concat(parts).sort_index()
    return out['close'].astype(float)


full = pd.DataFrame({s: load_full(s) for s in CORE})
fr = np.log(full).diff()
okf = full.index.to_series().diff() <= pd.Timedelta(hours=1)
for VT in (0.20, 0.30, 0.40, 0.50, 0.60):
    parts = []
    for s in CORE:
        c, rr = full[s], fr[s]
        M = np.log(c / c.shift(720))
        P = (np.sign(M) * (VT / (rr.rolling(720).std() * np.sqrt(HPY)))).clip(upper=3.0 * VT / 0.30).fillna(0.0)
        dpos = P.diff().abs().fillna(P.abs())
        parts.append((P.shift(1) * rr - dpos * RT / 2 / 1e4).where(okf).dropna())
    p = pd.concat(parts, axis=1).mean(axis=1).dropna()
    eq = (1 + p).cumprod()
    yrs = len(p) / HPY
    sh = p.mean() / p.std() * np.sqrt(HPY)
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    wr = ((1 + p).groupby(p.index.year).prod() - 1).min()
    print(f"  VT={VT:.0%}: CAGR={cagr:+5.0%}/год  Sharpe={sh:+.2f}  maxDD={dd:.0%}  "
          f"капитал x{eq.iloc[-1]:5.1f} за {yrs:.1f} г  худший год {wr:+.0%}")

# ---------- вердикт ----------
print("\n=== Критерии §2 (первичный XS) ===")
c1 = xs_sh >= 0.5
c2 = h1.sum() > 0 and h2.sum() > 0
c3 = xs_dd >= -0.50
print(f"1. Sharpe {xs_sh:+.2f} >= 0.5 -> {'ДА' if c1 else 'НЕТ'}")
print(f"2. H1 {h1.sum()*1e2:+.1f}% > 0 и H2 {h2.sum()*1e2:+.1f}% > 0 -> {'ДА' if c2 else 'НЕТ'}")
print(f"3. maxDD {xs_dd:.0%} >= -50% -> {'ДА' if c3 else 'НЕТ'}")
v1 = 'PROMOTE' if (c1 and c2 and c3) else 'REJECT'
print(f"\n=== Критерии §3 (вторичный альт K=168) ===")
d1 = ab_sh >= 0.5
d2 = pos_n >= 6
d3 = ab_dd >= -0.50
print(f"1. Корзина Sharpe {ab_sh:+.2f} >= 0.5 -> {'ДА' if d1 else 'НЕТ'}")
print(f"2. Альтов в плюсе: {pos_n}/8 (порог >= 6) -> {'ДА' if d2 else 'НЕТ'}")
print(f"3. maxDD корзины {ab_dd:.0%} >= -50% -> {'ДА' if d3 else 'НЕТ'}")
v2 = 'PROMOTE' if (d1 and d2 and d3) else 'REJECT'
print(f"\n=== ВЕРДИКТ E013: XS-рукав {v1}; альт-рукав {v2} ===")
