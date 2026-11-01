# Проверка знака кросс-сигнала ETH->BTC на свежих live-минутках
# (не бит-в-бит реплика e010: точные формулы e010 утеряны со старым чатом;
#  проверяем выжившую гипотезу: ETH возвращается к доходности BTC)
# Данные: официальные минутные бары Binance futures UM (data.binance.vision)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from zipfile import ZipFile

RAW = "/home/z/my-project/binfut-platform-restore/data/raw"
ALL_DATES = [f"2026-10-{d:02d}" for d in range(25, 32)]
import os
from zipfile import is_zipfile
# часть дневных архивов может быть ещё не выложена (XML-заглушка) — берём только валидные zip
DATES = [d for d in ALL_DATES
         if os.path.exists(f"{RAW}/BTCUSDT-1m-{d}.zip")
         and is_zipfile(f"{RAW}/BTCUSDT-1m-{d}.zip")
         and is_zipfile(f"{RAW}/ETHUSDT-1m-{d}.zip")]
COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qvol', 'count', 'tbvol', 'tbqvol', 'ig']


def load_sym(sym: str) -> pd.DataFrame:
    parts = []
    for d in DATES:
        with ZipFile(f"{RAW}/{sym}-1m-{d}.zip") as z:
            with z.open(z.namelist()[0]) as f:
                # часть архивов с заголовком, часть без — определяем по первой строке
                head = f.readline().decode()
                f.seek(0)
                has_header = head.lower().startswith('open_time')
                df = pd.read_csv(
                    f, header=0 if has_header else None, names=COLS,
                    usecols=['open_time', 'close'])
        df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
        parts.append(df.set_index('dt')[['close']])
    out = pd.concat(parts)
    assert out.index.is_unique, "дубликаты минут"
    assert out.index.size == len(DATES) * 1440, \
        f"ожидаю {len(DATES) * 1440} минут, есть {out.index.size}"
    return out


btc = load_sym('BTCUSDT')
eth = load_sym('ETHUSDT')
m = btc.join(eth, lsuffix='_btc', rsuffix='_eth', how='inner').dropna()
m['r_btc'] = np.log(m['close_btc']).diff()
m['r_eth'] = np.log(m['close_eth']).diff()

# Сигнал: насколько ETH ОТСТАЛ от BTC за прошлые W минут (BTC - ETH, кумулятивно)
WINDOWS = [30, 60, 120]
HORIZONS = [15, 30, 60, 120]
for w in WINDOWS:
    m[f'spread{w}'] = (m['r_btc'] - m['r_eth']).rolling(w).sum()
for h in HORIZONS:
    m[f'fwd{h}'] = m['r_eth'].rolling(h).sum().shift(-h)

print(f"минут в выборке: {len(m)}  ({m.index[0]} .. {m.index[-1]})")
print(f"{'окно':>4} {'гориз':>5} {'n':>5} {'IC':>8} {'p-value':>9}")
rows = []
for w in WINDOWS:
    for h in HORIZONS:
        d = m[[f'spread{w}', f'fwd{h}']].dropna()
        ic, p = spearmanr(d[f'spread{w}'], d[f'fwd{h}'])
        rows.append((w, h, len(d), ic, p))
        print(f"{w:>4} {h:>5} {len(d):>5} {ic:>8.4f} {p:>9.4g}")

# День за днём (устойчивость знака): окно 60, горизонт 60
print("\nДень за днём (окно 60, горизонт 60):")
for d in DATES:
    sub = m.loc[d]
    dd = sub[['spread60', 'fwd60']].dropna()
    if len(dd) > 30:
        ic, p = spearmanr(dd['spread60'], dd['fwd60'])
        print(f"  {d}: IC={ic:+.4f}  n={len(dd)}  p={p:.3g}")

ic_arr = np.array([r[3] for r in rows])
print(f"\nзнак IC: положительных {int((ic_arr > 0).sum())} из {len(ic_arr)}")
