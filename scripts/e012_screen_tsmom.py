# E012 — экспресс-скрин трендовой гипотезы (TSMOM) на DEV 2021-01..2025-12
# 1h бары, 2 символа. Не вердикт — отбор кандидата.
# Позиция: sign(доходность за K часов), вход по закрытию бара (лаг 1 бар),
# издержки: RT maker 4.5 bps на каждую смену стороны (4.5*|dpos|/2).
import numpy as np
import pandas as pd
from zipfile import ZipFile, is_zipfile

RAW = "/home/z/my-project/binfut-platform-restore/data/raw/klines1h"
YEARS = range(2021, 2026)
KCOLS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
         'close_time', 'qvol', 'count', 'tbvol', 'tbqvol', 'ig']
RT_BPS = 4.5
HOURS_PER_YEAR = 8760


def load_close_1h(sym):
    parts = []
    for y in YEARS:
        for m in range(1, 13):
            p = f"{RAW}/{sym}-1h-{y}-{m:02d}.zip"
            assert is_zipfile(p), f"нет {p}"
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
    assert out.index.is_unique
    return out['close'].astype(float)


def stats(r, pos, name):
    """r — часовая доходность, pos — позиция (эффективна с этого бара)."""
    dpos = pos.diff().abs().fillna(pos.abs())
    cost = dpos * RT_BPS / 2 / 1e4
    strat = pos.shift(1).fillna(0) * r - cost  # лаг 1 бар: вход по след. закрытию
    strat = strat.dropna()
    ann_ret = strat.mean() * HOURS_PER_YEAR
    ann_vol = strat.std() * np.sqrt(HOURS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    eq = (1 + strat).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    flips_per_year = dpos.sum() / 2 / (len(strat) / HOURS_PER_YEAR)
    print(f"  {name:<28} net={ann_ret:+7.1%}/год  Sharpe={sharpe:+.2f}  "
          f"maxDD={dd:.0%}  смен/год={flips_per_year:5.1f}")


for sym in ('BTCUSDT', 'ETHUSDT'):
    close = load_close_1h(sym)
    r = np.log(close).diff()
    print(f"\n### {sym}  ({close.index[0].date()} .. {close.index[-1].date()}, "
          f"{len(close)} часов) — net после 4.5 bps RT")
    for K in (24, 72, 168, 720):
        sig = np.log(close / close.shift(K))
        pos = np.sign(sig)
        stats(r, pos, f"TSMOM K={K}ч ({K//24}д)")
    # вариант с вол-таргетом (цель 30% годовых)
    vol = r.rolling(30 * 24).std() * np.sqrt(HOURS_PER_YEAR)
    for K in (72, 168):
        sig = np.log(close / close.shift(K))
        pos = np.sign(sig) * (0.30 / vol).clip(upper=3.0)
        stats(r, pos, f"TSMOM K={K}ч + волтаргет30%")

# портфель 50/50 BTC+ETH, K=168
c1, c2 = load_close_1h('BTCUSDT'), load_close_1h('ETHUSDT')
m = pd.DataFrame({'c1': c1, 'c2': c2}).dropna()
r1, r2 = np.log(m['c1']).diff(), np.log(m['c2']).diff()
for K in (72, 168):
    p1 = np.sign(m['c1'] / m['c1'].shift(K) - 1)
    p2 = np.sign(m['c2'] / m['c2'].shift(K) - 1)
    pos = 0.5 * p1 + 0.5 * p2
    rp = 0.5 * r1 + 0.5 * r2
    stats(rp, pos, f"ПОРТФЕЛЬ 50/50 K={K//24}д")
print("\n(скрин завершён; отбор — не вердикт)")
