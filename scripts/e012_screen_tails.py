# E012 — экспресс-скрин «хвостовой» версии кросс-реверсии ETH->BTC
# Идея из E011: непрерывный сигнал слаб, но что если входить ТОЛЬКО на
# экстремальных расхождениях (топ-10% |S|)? Меньше сделок, больше движение.
# DEV-данные 2025-07..12. Не вердикт. Порог для скрина — скользящий
# 30-дневный квантиль (без подглядывания вперёд).
import numpy as np
import pandas as pd
from zipfile import ZipFile, is_zipfile

RAW = "/home/z/my-project/binfut-platform-restore/data/raw"
MONTHS = [f"2025-{m:02d}" for m in range(7, 13)]
KCOLS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
         'close_time', 'qvol', 'count', 'tbvol', 'tbqvol', 'ig']


def load_close(sym):
    parts = []
    for m in MONTHS:
        with ZipFile(f"{RAW}/{sym}-1m-{m}.zip") as z:
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


btc, eth = load_close('BTCUSDT'), load_close('ETHUSDT')
m = pd.DataFrame({'btc': btc, 'eth': eth}).dropna()
m['r_btc'] = np.log(m['btc']).diff()
m['r_eth'] = np.log(m['eth']).diff()
W = 120
m['S'] = (m['r_btc'] - m['r_eth']).rolling(W).sum()  # >0: ETH отстал

# скользящий 30-дневный квантиль |S| (только прошлое)
absS = m['S'].abs()
thr = absS.rolling(30 * 1440, min_periods=5 * 1440).quantile(0.90)
m['event'] = absS > thr  # момент экстремального расхождения

print(f"минут: {len(m)}; событий (вход в топ-10% |S|): {int(m['event'].sum())}")
print(f"(события кластеризуются; уникальных эпизодов см. ниже)")

# События -> эпизоды (разрешаем новый вход не раньше чем через H минут)
results = []
for H in (60, 120, 240):
    in_pos = pd.Series(False, index=m.index)
    pos = pd.Series(np.nan, index=m.index)
    i = 0
    idx = m.index
    S = m['S'].values
    ev = m['event'].values
    r_eth = m['r_eth'].values
    n = len(m)
    # простая симуляция: в момент события входим sign(S) на H минут
    holding = 0
    direction = 0.0
    trades = []
    entry_ret_accum = 0.0
    for i in range(n):
        if holding > 0:
            trades[-1] += r_eth[i] * direction
            holding -= 1
        elif ev[i]:
            direction = 1.0 if S[i] > 0 else -1.0
            trades.append(0.0)
            holding = H
    trades = np.array(trades)
    # издержки: RT maker 4.5 bps на сделку (вход+выход)
    net_bps = trades * 1e4 - 4.5
    # эпизоды: события группируются; считаем и уникальные окна
    ev_starts = m['event'].values.copy()
    # число неперекрывающихся эпизодов ~ число сделок
    print(f"\nH={H} мин: сделок={len(trades)} | "
          f"gross ср={trades.mean()*1e4:+.2f} bps | "
          f"net(4.5RT) ср={net_bps.mean():+.2f} bps | "
          f"медиана net={np.median(net_bps):+.2f} | "
          f"hit-rate={(net_bps > 0).mean():.0%} | "
          f"сумма net={net_bps.sum():+.0f} bps")

# то же для ETH-стороны (сигнал применяем к ETH) — уже так; теперь BTC-сторона:
print("\n(сигнал: long ETH при S>0 [ETH отстал], short ETH при S<0)")
