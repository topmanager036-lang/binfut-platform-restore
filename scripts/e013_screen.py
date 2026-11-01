# E013 — dev-скрин (2021-01..2025-12). ДО просмотра holdout.
# Семейство A: TSMOM по-символьно, K in {24,72,168,360,720} — двигатель E012
#   (VT30, clip 3, лаг 1 бар, maker 4.5 bps RT) на 8 альтах + BTC/ETH.
# Семейство B: кросс-секционный моментум (lookback 720ч, ребаланс 24/168ч,
#   лонг топ-3 / шорт боттом-3, слот = (0.30/vol_i)/6, clip 1.0).
# Семейство C: комбо. Все цифры — DEV, не вердикты.
import numpy as np
import pandas as pd
from zipfile import ZipFile, is_zipfile
import os

RAW = "/home/z/my-project/binfut-platform-restore/data/raw/klines1h"
KCOLS = ['open_time', 'close']
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT"]
ALTS = SYMS[2:]
HPY = 8760
RT = 4.5   # bps round-trip maker
DEV_START, DEV_END = "2021-01-01", "2025-12-31 23:59"


def load_1h(sym, y0=2021, y1=2025):
    parts = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            p = f"{RAW}/{sym}-1h-{y}-{m:02d}.zip"
            if not is_zipfile(p):
                continue
            with ZipFile(p) as z:
                with z.open(z.namelist()[0]) as f:
                    head = f.readline().decode()
                    f.seek(0)
                    df = pd.read_csv(f, header=0 if head.lower().startswith('open_time') else None,
                                     names=['open_time', 'open', 'high', 'low', 'close',
                                            'volume', 'ct', 'qvol', 'n', 'tbv', 'tbq', 'ig'],
                                     usecols=[0, 4])
            parts.append(df.set_index(pd.to_datetime(df['open_time'], unit='ms')))
    out = pd.concat(parts).sort_index()
    assert out.index.is_unique
    return out['close'].astype(float)


close = pd.DataFrame({s: load_1h(s) for s in SYMS}).loc[DEV_START:DEV_END]
print(f"панель dev: {close.index[0]} .. {close.index[-1]}, {len(close)} ч; "
      f"не-NaN по символам: " + "/".join(str(int(close[s].notna().sum() // 24)) + "д" for s in SYMS))

r = np.log(close).diff()
vol = r.rolling(720).std() * np.sqrt(HPY)
step_ok = close.index.to_series().diff() <= pd.Timedelta(hours=1)


def tsmom(sym, K):
    c, rr = close[sym], r[sym]
    M = np.log(c / c.shift(K))
    P = (np.sign(M) * (0.30 / vol[sym])).clip(upper=3.0).fillna(0.0)
    dpos = P.diff().abs().fillna(P.abs())
    strat = (P.shift(1) * rr - dpos * RT / 2 / 1e4)
    strat = strat.where(step_ok)
    return strat.dropna(), dpos


print("\n=== A. TSMOM по-символьно (net %/год, Sharpe; VT30, maker 4.5 bps RT, лаг 1) ===")
print(f"{'K':>6} | " + " | ".join(f"{s.replace('USDT',''):>7}" for s in SYMS) + " | альт-корз | 10-симв")
baskets = {}
for K in [24, 72, 168, 360, 720]:
    cells, streams = [], []
    for s in SYMS:
        st, _ = tsmom(s, K)
        ann = st.mean() * HPY
        sh = st.mean() / st.std() * np.sqrt(HPY)
        cells.append(f"{ann:+5.0%}/{sh:+.2f}")
        streams.append(st)
    panel = pd.concat(streams, axis=1)
    alt_b = panel[ALTS].mean(axis=1).dropna()
    all_b = panel.mean(axis=1).dropna()
    for nm, ser in [("alt", alt_b), ("all", all_b)]:
        baskets[(K, nm)] = ser
    fa = lambda x: f"{x.mean()*HPY:+.0%}/г S{x.mean()/x.std()*np.sqrt(HPY):+.2f}"
    print(f"{K:>6} | " + " | ".join(cells) + f" | {fa(alt_b):>9} | {fa(all_b):>9}")

print("\n=== B. Кросс-секционный моментум (LB=720ч; лонг топ-3/шорт боттом-3; лаг 1ч) ===")
xs_streams = {}
for R in [24, 168]:
    L = np.log(close / close.shift(720))
    n = len(close)
    rebal = np.zeros(n, bool)
    rebal[::R] = True
    W = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    last_w = pd.Series(0.0, index=close.columns)
    # W: NaN везде; ПОЛНАЯ строка весов (с нулями для невыбранных) в точки ребаланса,
    # тогда ffill корректно обнуляет выбывшие из топ/боттом символы.
    W = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    Li = L.values
    Ci = close.columns.tolist()
    for i in np.where(rebal)[0]:
        if i < 720:
            continue
        row = pd.Series(Li[i], index=Ci).dropna()
        if len(row) < 6:
            continue
        ranks = row.rank(ascending=False)
        w = pd.Series(0.0, index=Ci)
        longs, shorts = ranks.head(3).index, ranks.tail(3).index
        for s in longs:
            w[s] = +min((0.30 / vol[s].iloc[i]) / 6, 1.0)
        for s in shorts:
            w[s] = -min((0.30 / vol[s].iloc[i]) / 6, 1.0)
        W.iloc[i] = w.values
    Wf = W.ffill().fillna(0.0)
    turn = Wf.diff().abs().sum(axis=1).iloc[1:]
    turn.iloc[0] = Wf.iloc[0].abs().sum()
    strat = (Wf.shift(1) * r.fillna(0.0)).sum(axis=1) - turn * RT / 2 / 1e4
    strat = strat.where(step_ok).dropna()
    xs_streams[R] = strat
    sh = strat.mean() / strat.std() * np.sqrt(HPY)
    eq = (1 + strat).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    swaps = int((Wf.diff().abs().sum(axis=1) > 0).sum())
    pvol = strat.std() * np.sqrt(HPY)
    print(f"R={R:>3}ч: net={strat.mean()*HPY:+.1%}/год  Sharpe={sh:+.2f}  maxDD={dd:.0%}  "
          f"вол={pvol:.0%}  ср.гросс={Wf.abs().sum(axis=1).mean():.2f}  "
          f"часов с ребалансом={swaps} (только смена состава/весов)")

print("\n=== C. Комбо (EW, одна строка = один блок портфеля) ===")
core = pd.concat([tsmom('BTCUSDT', 720)[0], tsmom('ETHUSDT', 720)[0]], axis=1).mean(axis=1)
altb = baskets[(720, "alt")]
xs = xs_streams[168]
for name, parts in [("core BTC/ETH K720", [core]),
                    ("альт-корзина K720", [altb]),
                    ("XS R=168", [xs]),
                    ("core+альты (EW)", [core, altb]),
                    ("core+альты+XS (EW)", [core, altb, xs])]:
    p = pd.concat(parts, axis=1).mean(axis=1).dropna()
    sh = p.mean() / p.std() * np.sqrt(HPY)
    eq = (1 + p).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = len(p) / HPY
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    print(f"{name:>22}: CAGR={cagr:+.0%}  Sharpe={sh:+.2f}  maxDD={dd:.0%}  "
          f"капитал x{eq.iloc[-1]:.1f} за {yrs:.1f} г")

print("\n=== Погодово (портфель, % за год) ===")
for name, ser in [("альт-корз K720", altb), ("XS R=168", xs),
                  ("core+альты+XS", pd.concat([core, altb, xs], axis=1).mean(axis=1))]:
    y = (1 + ser).groupby(ser.index.year).prod() - 1
    print(f"{name:>16}: " + "  ".join(f"{yr}: {v:+.0%}" for yr, v in y.items()))
