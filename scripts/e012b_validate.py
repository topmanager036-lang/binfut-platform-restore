# E012b — расширенная валидация по плану docs/E012B_PLAN.md
# Стратегия = E012 primary (K=720, волтаргет 30%, clip 3, лаг 1 бар, maker 4.5 bps RT)
# Полная история 2020-01..2026-10 (P&L с warmup ~2 мес, т.е. с марта 2020).
import numpy as np
import pandas as pd
from zipfile import ZipFile, is_zipfile

RAW = "/home/z/my-project/binfut-platform-restore/data/raw/klines1h"
KCOLS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
         'close_time', 'qvol', 'count', 'tbvol', 'tbqvol', 'ig']
SYMS = ['BTCUSDT', 'ETHUSDT']
HPY = 8760
K = 720
VT = 0.30
RT = 4.5


def load_1h(sym):
    files = [f"{RAW}/{sym}-1h-2020-{m:02d}.zip" for m in range(1, 13)]
    files += [f"{RAW}/{sym}-1h-{y}-{m:02d}.zip" for y in range(2021, 2026)
              for m in range(1, 13)]
    files += [f"{RAW}/{sym}-1h-2026-{m:02d}.zip" for m in range(1, 10)]
    files += [f"{RAW}/{sym}-1h-2026-10-{d:02d}.zip" for d in range(1, 32)]
    parts = []
    for p in files:
        if not is_zipfile(p):
            continue
        with ZipFile(p) as z:
            with z.open(z.namelist()[0]) as f:
                head = f.readline().decode()
                f.seek(0)
                has_header = head.lower().startswith('open_time')
                df = pd.read_csv(f, header=0 if has_header else None,
                                 names=KCOLS, usecols=['open_time', 'close'])
        parts.append(df.set_index(
            pd.to_datetime(df['open_time'], unit='ms'))[['close']])
    out = pd.concat(parts)
    assert out.index.is_unique, f"{sym}: дубликаты"
    return out['close'].astype(float)


m = pd.DataFrame({s: load_1h(s) for s in SYMS}).dropna()
r = np.log(m).diff()
step = m.index.to_series().diff()
valid = step <= pd.Timedelta(hours=1)
print(f"панель: {len(m)} часов, {m.index[0]} .. {m.index[-1]}; "
      f"gap-исключений: {int((~valid).sum())}")


def arm(strat_r, close, K, vt, rt):
    M = np.log(close / close.shift(K))
    vol = strat_r.rolling(720).std() * np.sqrt(HPY)
    P = (np.sign(M) * (vt / vol).clip(upper=3.0)).fillna(0.0)
    cost = P.diff().abs().fillna(P.abs()) * rt / 2 / 1e4
    s = (P.shift(1) * strat_r - cost).where(valid)
    return s.dropna()


strats = {s: arm(r[s], m[s], K, VT, RT) for s in SYMS}
port = (strats['BTCUSDT'] + strats['ETHUSDT']) / 2
bh = (r['BTCUSDT'] + r['ETHUSDT']).where(valid).dropna() / 2  # B&H 50/50


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(HPY) if x.std() > 0 else np.nan


def maxdd(x):
    eq = (1 + x).cumprod()
    return (eq / eq.cummax() - 1).min()


print("\n=== 1) Полная история (портфель 50/50, maker 4.5 bps RT) ===")
yrs = len(port) / HPY
g_eq = float((1 + port).prod() - 1)
g_ann = (1 + g_eq) ** (1 / yrs) - 1
bh_eq = float(np.exp(bh.sum()) - 1)
print(f"стратегия: рост капитала {g_eq:+.0%} за {yrs:.1f} г "
      f"(сложный {g_ann:+.1%}/год), Sharpe={sharpe(port):+.2f}, "
      f"maxDD={maxdd(port):.0%}")
print(f"buy-hold:  рост капитала {bh_eq:+.0%}, Sharpe={sharpe(bh):+.2f}, "
      f"maxDD={maxdd(bh):.0%}")
for s in SYMS:
    s_ann = float((1 + strats[s]).prod() ** (HPY / len(strats[s])) - 1)
    print(f"  {s}: {s_ann:+.1%}/год Sharpe={sharpe(strats[s]):+.2f} "
          f"maxDD={maxdd(strats[s]):.0%}")

print("\n=== 2) Погодовая таблица (портфель; B&H справочно) ===")
print(f"{'год':>4} | {'стратегия net':>13} | {'Sharpe':>7} | {'maxDD':>6} | {'B&H net':>8}")
for y in range(2020, 2027):
    py = port[port.index.year == y]
    by = bh[bh.index.year == y]
    if len(py) < 500:
        continue
    py_tot = float((1 + py).prod() - 1)
    by_tot = float(np.exp(by.sum()) - 1)
    print(f"{y:>4} | {py_tot:>12.1%} | {sharpe(py):>7.2f} | {maxdd(py):>6.0%} "
          f"| {by_tot:>8.1%}")

print("\n=== 3) Помесячная статистика (портфель) ===")
pm = (1 + port).groupby(port.index.to_period('M')).prod() - 1
print(f"месяцев: {len(pm)}; в плюсе: {int((pm > 0).sum())} "
      f"({(pm > 0).mean():.0%})")
worst = pm.nsmallest(5)
best = pm.nlargest(5)
print("худшие 5: " + ", ".join(f"{i} {v:+.1%}" for i, v in worst.items()))
print("лучшие 5: " + ", ".join(f"{i} {v:+.1%}" for i, v in best.items()))

print("\n=== 4) Медвежьи фокусы ===")
covid = port.loc['2020-03-01':'2020-04-30']
print(f"COVID (мар-апр 2020): net={float((1+covid).prod()-1):+.1%}, "
      f"Sharpe={sharpe(covid):+.2f}")
y22 = port[port.index.year == 2022]
b22 = bh[bh.index.year == 2022]
print(f"2022 (медвежий): net={float((1+y22).prod()-1):+.1%} "
      f"(B&H {float(np.exp(b22.sum())-1):+.1%}), "
      f"Sharpe={sharpe(y22):+.2f}, maxDD={maxdd(y22):.0%}")
h1_2024 = port.loc['2024-01-01':'2024-06-30']
print(f"2024 H1 (чоп): net={float((1+h1_2024).prod()-1):+.1%}, "
      f"Sharpe={sharpe(h1_2024):+.2f}")

print("\n=== 5) Чувствительность (отчёт, не подбор) ===")
print("K-сетка (портфель, волтаргет 30%):")
for k in (360, 480, 600, 720, 840, 960, 1080):
    p = sum(arm(r[s], m[s], k, VT, RT) for s in SYMS) / 2
    k_ann = float((1 + p).prod() ** (HPY / len(p)) - 1)
    print(f"  K={k:>4}ч ({k//24:>2}д): {k_ann:>7.1%}/год "
          f"Sharpe={sharpe(p):+.2f} maxDD={maxdd(p):.0%}")
print("волтаргет-сетка (портфель, K=720):")
for vt in (0.20, 0.30, 0.40):
    p = sum(arm(r[s], m[s], K, vt, RT) for s in SYMS) / 2
    vt_ann = float((1 + p).prod() ** (HPY / len(p)) - 1)
    print(f"  VT={vt:.0%}: {vt_ann:>7.1%}/год Sharpe={sharpe(p):+.2f} "
          f"maxDD={maxdd(p):.0%}")
