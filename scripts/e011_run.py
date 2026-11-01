# E011 — прогон по pre-registration (docs/E011_PREREGISTRATION.md)
# Holdout: 2025-07..12, НИ РАЗУ не просматривался до этого прогона.
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from zipfile import ZipFile, is_zipfile

RAW = "/home/z/my-project/binfut-platform-restore/data/raw"
MONTHS = [f"2025-{m:02d}" for m in range(7, 13)]
COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qvol', 'count', 'tbvol', 'tbqvol', 'ig']

# ---- критерии из протокола (скопированы ДОЛЬСЯ точно) ----
W_MAIN, H_MAIN = 120, 60
IC_MIN, P_MAX = 0.03, 0.01
WEEK_SHARE_MIN = 0.60
RT_MAKER_BPS, RT_TAKER_BPS = 4.5, 15.0


def load_sym(sym: str) -> pd.DataFrame:
    parts = []
    for m in MONTHS:
        p = f"{RAW}/{sym}-1m-{m}.zip"
        assert os.path.exists(p) and is_zipfile(p), f"нет архива {p}"
        with ZipFile(p) as z:
            with z.open(z.namelist()[0]) as f:
                head = f.readline().decode()
                f.seek(0)
                has_header = head.lower().startswith('open_time')
                df = pd.read_csv(f, header=0 if has_header else None,
                                 names=COLS, usecols=['open_time', 'close'])
        df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
        parts.append(df.set_index('dt')[['close']])
    out = pd.concat(parts)
    assert out.index.is_unique, "дубликаты минут"
    return out


btc = load_sym('BTCUSDT')
eth = load_sym('ETHUSDT')
m = btc.join(eth, lsuffix='_btc', rsuffix='_eth', how='inner').dropna()

print("=== Контроль качества данных ===")
print(f"минут: {len(m)} (ожидалось ~{6*43200}); "
      f"период {m.index[0]} .. {m.index[-1]}")
gaps = m.index.to_series().diff().dt.total_seconds().div(60)
n_gaps = int((gaps > 1).sum())
print(f"пропусков (шаг > 1 мин): {n_gaps}")

m['r_btc'] = np.log(m['close_btc']).diff()
m['r_eth'] = np.log(m['close_eth']).diff()
m['S'] = (m['r_btc'] - m['r_eth']).rolling(W_MAIN).sum()
m['fwd'] = m['r_eth'].rolling(H_MAIN).sum().shift(-H_MAIN)

# ---------- 1) Основной критерий: IC ----------
d = m[['S', 'fwd']].dropna()
ic, p = spearmanr(d['S'], d['fwd'])
print("\n=== 1) Основная конфигурация W=120, h=60 ===")
print(f"IC = {ic:+.4f}   p = {p:.3g}   n = {len(d)}")
c1 = (ic >= IC_MIN) and (p < P_MAX)
print(f"Критерий 1 (IC>={IC_MIN} и p<{P_MAX}): {'ВЫПОЛНЕН' if c1 else 'НЕ ВЫПОЛНЕН'}")

# ---------- 2) Понедельная устойчивость ----------
d2 = d.copy()
d2['week'] = d2.index.to_period('W')
weekly = d2.groupby('week').apply(
    lambda g: spearmanr(g['S'], g['fwd']).statistic if len(g) > 100 else np.nan,
    include_groups=False).dropna()
share = float((weekly > 0).mean())
print("\n=== 2) Понедельная разбивка ===")
print(f"недель: {len(weekly)}; с IC>0: {int((weekly > 0).sum())} "
      f"({share:.0%}); медиана IC={weekly.median():+.4f}")
c2 = share >= WEEK_SHARE_MIN
print(f"Критерий 2 (>= {WEEK_SHARE_MIN:.0%} недель с IC>0): "
      f"{'ВЫПОЛНЕН' if c2 else 'НЕ ВЫПОЛНЕН'}")

# ---------- 3) Экономика: позиция = знак S, исполнение next-minute ----------
pos = np.sign(m['S']).shift(1)  # решение в конце t, позиция действует в t+1
ret = (pos * m['r_eth']).dropna()
turn = pos.diff().abs().dropna()  # единицы оборота на смену позиции
gross_bps = ret.sum() * 1e4
for name, rt in [('maker', RT_MAKER_BPS), ('taker (стресс)', RT_TAKER_BPS)]:
    cost_bps = turn.sum() * rt / 2
    net_bps = gross_bps - cost_bps
    print(f"\n=== 3) Симуляция, издержки {name} RT {rt} bps ===")
    print(f"gross = {gross_bps:+.1f} bps | издержки = {cost_bps:.1f} bps "
          f"| NET = {net_bps:+.1f} bps")
    print(f"время в позиции: {(pos != 0).mean():.0%}; "
          f"смен стороны: {int((turn > 0).sum())}; "
          f"оборот (ед.): {turn.sum():.0f}")
    if name == 'maker':
        c3 = net_bps > 0
        print(f"Критерий 3 (net maker > 0): {'ВЫПОЛНЕН' if c3 else 'НЕ ВЫПОЛНЕН'}")

# ---------- Чувствительность (отчёт, не решение) ----------
print("\n=== Чувствительность (отчёт) ===")
print(f"{'окно':>4} {'гориз':>5} {'IC':>8}")
for w in [30, 60, 120]:
    for h in [15, 30, 60, 120]:
        s = (m['r_btc'] - m['r_eth']).rolling(w).sum()
        fw = m['r_eth'].rolling(h).sum().shift(-h)
        dd = pd.concat([s, fw], axis=1, keys=['s', 'f']).dropna()
        icx, _ = spearmanr(dd['s'], dd['f'])
        print(f"{w:>4} {h:>5} {icx:>8.4f}")

verdict = 'PROMOTE' if (c1 and c2 and c3) else 'REJECT'
print(f"\n=== ВЕРДИКТ E011 (holdout 2025-07..12): {verdict} ===")
