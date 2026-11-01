# E012 — прогон по pre-registration (docs/E012_PREREGISTRATION.md)
# Holdout: 2026-01-01..2026-10-31 (минус 10-28), НЕ просматривался до прогона.
import hashlib
import numpy as np
import pandas as pd
from zipfile import ZipFile, is_zipfile

RAW = "/home/z/my-project/binfut-platform-restore/data/raw/klines1h"
KCOLS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
         'close_time', 'qvol', 'count', 'tbvol', 'tbqvol', 'ig']
SYMS = ['BTCUSDT', 'ETHUSDT']
ARMS = {'primary': 720, 'secondary': 72}
RT_BPS = {'maker': 4.5, 'taker': 15.0}
HPY = 8760  # часов в году
PANEL_START = '2025-12-01'   # warmup с декабря 2025 (dev; P&L не считается)
PNL_START = '2026-01-01 00:00:00'


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_1h(sym: str) -> pd.Series:
    files = []
    # месячные 2025-12..2026-09 + дневные 2026-10
    files += [f"{RAW}/{sym}-1h-2025-12.zip"]
    files += [f"{RAW}/{sym}-1h-2026-{m:02d}.zip" for m in range(1, 10)]
    files += [f"{RAW}/{sym}-1h-2026-10-{d:02d}.zip" for d in range(1, 32)]
    parts = []
    used = []
    for p in files:
        if not (is_zipfile(p)):
            continue  # 2026-10-28 не опубликован — известный пропуск
        used.append(p)
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
    assert out.index.is_unique, f"{sym}: дубликаты часов"
    return out['close'].astype(float), used


data, used_files = {}, []
for sym in SYMS:
    data[sym], uf = load_1h(sym)
    used_files += uf
print("=== Контроль данных ===")
digests = {p: sha256(p) for p in sorted(set(used_files))}
print(f"архивов: {len(digests)}; пропущен 2026-10-28 (не опубликован биржей)")
print(f"примеры SHA256: BTC 2026-01: {digests[f'{RAW}/BTCUSDT-1h-2026-01.zip']}"
      f"; ETH 2026-01: {digests[f'{RAW}/ETHUSDT-1h-2026-01.zip']}")

m = pd.DataFrame({s: data[s] for s in SYMS}).loc[PANEL_START:].dropna()
r = np.log(m).diff()
# часы с разрывом > 1ч исключаются из P&L (регистрация §2)
step = m.index.to_series().diff()
valid = step <= pd.Timedelta(hours=1)
n_gap = int((~valid).sum())
print(f"часов панели: {len(m)} ({m.index[0]} .. {m.index[-1]}); "
      f"разрывов-исключений: {n_gap}")


def arm_pnl(sym_r: pd.Series, close: pd.Series, K: int, rt: float):
    M = np.log(close / close.shift(K))
    D = np.sign(M)
    vol = sym_r.rolling(720).std() * np.sqrt(HPY)
    P = (D * (0.30 / vol).clip(upper=3.0)).fillna(0.0)
    dpos = P.diff().abs().fillna(P.abs())
    cost = dpos * rt / 2 / 1e4
    strat = (P.shift(1) * sym_r - cost)
    strat = strat.where(valid)          # P&L только на валидных часах
    return strat.dropna(), dpos


def report(name, strat, quiet=False):
    ann = strat.mean() * HPY
    sharpe = strat.mean() / strat.std() * np.sqrt(HPY) if strat.std() > 0 else np.nan
    eq = (1 + strat).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    if not quiet:
        print(f"  {name:<30} net={ann:+8.1%}/год  Sharpe={sharpe:+.2f}  maxDD={dd:.0%}")
    return ann, sharpe, dd


holdout = slice(PNL_START, None)
results = {}
for arm, K in ARMS.items():
    print(f"\n=== Рукав {arm}: K={K}ч, издержки maker 4.5 bps RT ===")
    per_sym, port_parts = {}, []
    for sym in SYMS:
        strat, dpos = arm_pnl(r[sym], m[sym], K, RT_BPS['maker'])
        strat_h = strat.loc[holdout]
        ann, sh, dd = report(sym, strat_h)
        per_sym[sym] = (ann, sh, dd, dpos.loc[holdout].sum() / 2)
        port_parts.append(strat_h)
    port = sum(port_parts) / len(port_parts)
    ann, sh, dd = report("ПОРТФЕЛЬ 50/50", port)
    results[arm] = {'per_sym': per_sym, 'port': (ann, sh, dd)}

    # стресс taker — только отчёт
    for sym in SYMS:
        strat_t, _ = arm_pnl(r[sym], m[sym], K, RT_BPS['taker'])
        report(f"  [taker стресс] {sym}", strat_t.loc[holdout], quiet=False)

print("\n=== Критерии вердикта (первичный рукав K=720, регистрация §5) ===")
p = results['primary']
c1 = p['port'][1] >= 0.5
c2 = all(p['per_sym'][s][0] > 0 for s in SYMS)
c3 = p['port'][2] >= -0.50
print(f"1. Net Sharpe портфеля >= 0.5:  {p['port'][1]:+.2f}  -> "
      f"{'ВЫПОЛНЕН' if c1 else 'НЕ ВЫПОЛНЕН'}")
for s in SYMS:
    print(f"2. {s}: net {p['per_sym'][s][0]:+.1%}/год -> "
          f"{'ок' if p['per_sym'][s][0] > 0 else 'ПРОВАЛ'}")
print(f"3. maxDD портфеля <= 50%:  {p['port'][2]:.0%}  -> "
      f"{'ВЫПОЛНЕН' if c3 else 'НЕ ВЫПОЛНЕН'}")
verdict = 'PROMOTE' if (c1 and c2 and c3) else 'REJECT'
print(f"\n=== ВЕРДИКТ E012 (holdout 2026-01..10): {verdict} ===")

sec = results['secondary']
sec_ok = sec['port'][1] >= 0.5 and all(sec['per_sym'][s][0] > 0 for s in SYMS) \
    and sec['port'][2] >= -0.50
print(f"(вторичный рукав K=72: Sharpe={sec['port'][1]:+.2f}, "
      f"net={sec['port'][0]:+.1%}, maxDD={sec['port'][2]:.0%} -> "
      f"{'критерии выполнены' if sec_ok else 'критерии НЕ выполнены'} — "
      f"в вердикт не входит)")
