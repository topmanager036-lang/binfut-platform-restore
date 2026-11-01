# E012b — помесячная статистика сетки волтаргета (калибровка ожиданий владельца)
# Полная история 2020-01..2026-10 (6.8 лет), IN-SAMPLE отчёт (см. журнал §15).
# Отвечает на вопрос: как выглядят РЕАЛЬНЫЕ месяцы стратегии E012 на каждом плече.
import numpy as np
import pandas as pd
from zipfile import ZipFile, is_zipfile

RAW = "/home/z/my-project/binfut-platform-restore/data/raw/klines1h"
HPY = 8760
RT = 4.5
CORE = ['BTCUSDT', 'ETHUSDT']


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
    return pd.concat(parts).sort_index()['close'].astype(float)


full = pd.DataFrame({s: load_full(s) for s in CORE})
fr = np.log(full).diff()
okf = full.index.to_series().diff() <= pd.Timedelta(hours=1)

print("Помесячная статистика E012 core (полная история 2020-01..2026-10, 81 месяц):")
print(f"{'VT':>4} | {'ср.мес':>7} {'медиана':>7} {'лучший':>7} {'худший':>7} "
      f"{'+% мес':>6} | {'худш. 3мес':>9} | CAGR/год")
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
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    m = (1 + p).groupby(p.index.to_period('M')).prod() - 1
    q = (1 + p).groupby(p.index.to_period('Q')).prod() - 1  # для худшей четверти
    print(f"{VT:>4.0%} | {m.mean():>+7.1%} {m.median():>+7.1%} {m.max():>+7.1%} "
          f"{m.min():>+7.1%} {int((m > 0).mean() * 100):>5}% | {q.min():>+9.1%} | {cagr:+.0%}")

print("\nЦелевая калибровка владельца: +20%/мес = +240%/год простыми = x8.9/год сложными.")
for label, sh in [("in-sample полная история", 1.53), ("честный holdout", 1.28)]:
    need_vol = 2.40 / sh
    print(f"  при Sharpe {sh} ({label}) нужна волатильность {need_vol:.0%}/год "
          f"(средняя позиция ~{need_vol / 0.45:.1f}x ноционально на символ)")
