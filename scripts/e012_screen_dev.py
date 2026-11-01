# E012 — экспресс-просеивание кандидатов на DEV-данных (2025-07..12)
# ВАЖНО: это исследовательский скрин, НЕ вердикт. Holdout для E012: 2026-01..06
# (ещё не скачан и не просматривался). Критерии будут зарегистрированы отдельно.
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from zipfile import ZipFile, is_zipfile

RAW = "/home/z/my-project/binfut-platform-restore/data/raw"
MONTHS = [f"2025-{m:02d}" for m in range(7, 13)]
KCOLS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
         'close_time', 'qvol', 'count', 'tbvol', 'tbqvol', 'ig']
SYMS = ['BTCUSDT', 'ETHUSDT']


def load_klines(subdir: str, sym: str, usecol: str = 'close') -> pd.Series:
    parts = []
    for m in MONTHS:
        p = f"{RAW}/{subdir}/{sym}-1m-{m}.zip" if subdir != '.' \
            else f"{RAW}/{sym}-1m-{m}.zip"
        assert is_zipfile(p), f"нет {p}"
        with ZipFile(p) as z:
            with z.open(z.namelist()[0]) as f:
                head = f.readline().decode()
                f.seek(0)
                has_header = head.lower().startswith('open_time') or \
                    head.lower().startswith('calc_time')
                df = pd.read_csv(f, header=0 if has_header else None,
                                 names=KCOLS, usecols=['open_time', usecol])
        s = df.set_index(pd.to_datetime(df['open_time'], unit='ms'))[usecol]
        parts.append(s)
    out = pd.concat(parts)
    assert out.index.is_unique
    return out.astype(float)


def load_funding(sym: str) -> pd.Series:
    parts = []
    for m in MONTHS:
        p = f"{RAW}/funding/{sym}-fundingRate-{m}.zip"
        with ZipFile(p) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
        idx = pd.to_datetime(df['calc_time'].round(-3) / 1e3, unit='s')
        parts.append(pd.Series(df['last_funding_rate'].values, index=idx))
    out = pd.concat(parts)
    out = out[~out.index.duplicated()]
    return out.astype(float)


print("=" * 68)
print("DEV-СКРИН E012 (2025-07..12). Только для отбора кандидатов.")
print("=" * 68)

for sym in SYMS:
    close = load_klines('.', sym)
    r = np.log(close).diff()
    fund = load_funding(sym)

    # ---------- A. Funding: контрарианская гипотеза + carry ----------
    print(f"\n### {sym} — A) FUNDING (события каждые 8ч, n={len(fund)})")
    fr = fund.copy()
    for h in (8, 24, 48):
        fwd = close.shift(-h * 60) / close - 1.0  # fwd return c момента события
        # пересчёт на минутный индекс: берём fwd в момент события
        fwd_min = fwd.reindex(fr.index, method='nearest', tolerance='1min')
        d = pd.concat([fr, fwd_min], axis=1, keys=['f', 'fwd']).dropna()
        ic, p = spearmanr(d['f'], d['fwd'])
        print(f"  fwd {h:>2}h: IC={ic:+.4f} p={p:.3g}")
    fwd8 = (close.shift(-8 * 60) / close - 1.0).reindex(
        fr.index, method='nearest', tolerance='1min')
    d = pd.concat([fr, fwd8], axis=1, keys=['f', 'fwd']).dropna()
    q = pd.qcut(d['f'], 5, labels=False, duplicates='drop')
    means = d.groupby(q)['fwd'].mean() * 1e4
    print("  квинтили funding -> ср. fwd 8h (bps):",
          " ".join(f"Q{i+1}:{v:+.1f}" for i, v in means.items()))
    neg = d[d['f'] < 0]['fwd'].mean() * 1e4
    pos = d[d['f'] >= 0]['fwd'].mean() * 1e4
    # carry: лонг при funding<0 зарабатывает ставку
    carry = d[d['f'] < 0].apply(
        lambda x: x['fwd'] + abs(x['f']), axis=1).mean() * 1e4
    print(f"  funding<0: fwd8={neg:+.1f} bps (n={int((d['f']<0).sum())}); "
          f"funding>=0: {pos:+.1f} bps")
    print(f"  carry-лонг при funding<0 (цена+ставка): {carry:+.1f} bps/8ч")

    # ---------- B. Premium mean-reversion ----------
    prem = load_klines('premium', sym)  # premium index (rate)
    pbps = prem * 1e4
    s60 = pbps.rolling(60).mean()
    print(f"\n### {sym} — B) PREMIUM (1ч среднее, bps; "
          f"медиана={pbps.median():+.2f}, std={pbps.std():.2f})")
    for h in (60, 240):
        fwd = r.rolling(h).sum().shift(-h)
        dd = pd.concat([s60, fwd], axis=1, keys=['s', 'fwd']).dropna()
        ic, p = spearmanr(dd['s'], dd['fwd'])
        print(f"  fwd {h//60}ч: IC={ic:+.4f} p={p:.3g} n={len(dd)}")
    fwd1 = r.rolling(60).sum().shift(-60)
    dd = pd.concat([s60, fwd1], axis=1, keys=['s', 'fwd']).dropna()
    q = pd.qcut(dd['s'], 3, labels=False, duplicates='drop')
    means = dd.groupby(q)['fwd'].mean() * 1e4
    print("  терцили premium -> ср. fwd 1ч (bps):",
          " ".join(f"T{i+1}:{v:+.2f}" for i, v in means.items()))
    # сама реверсия премии (не цена): IC premium vs fwd premium
    fwd_prem = pbps.rolling(60).mean().shift(-60)
    ddp = pd.concat([s60, fwd_prem], axis=1, keys=['s', 'fp']).dropna()
    icp, pp = spearmanr(ddp['s'], ddp['fp'])
    print(f"  реверсия самой премии: IC={icp:+.4f} p={pp:.3g} "
          f"(премия возвращается, но это не цена)")

    # ---------- C. Внутрисуточная сезонность ----------
    print(f"\n### {sym} — C) СЕЗОННОСТЬ")
    tod = r.groupby(r.index.hour).agg(['mean', lambda x: x.abs().mean()])
    tod.columns = ['mean_bps', 'vol_bps']
    tod['mean_bps'] *= 1e4
    tod['vol_bps'] *= 1e4
    print("  волатильность по часам UTC (bps/мин): мин="
          f"{tod['vol_bps'].idxmin()}ч ({tod['vol_bps'].min():.2f}), макс="
          f"{tod['vol_bps'].idxmax()}ч ({tod['vol_bps'].max():.2f})")
    # эффект времени фандинга 00/08/16 UTC
    mask_fund = r.index.minute.isin(range(0, 5)) & r.index.hour.isin([0, 8, 16])
    pre = r[(r.index.minute.isin(range(55, 60))) &
            (r.index.hour.isin([23, 7, 15]))]
    post = r[mask_fund]
    print(f"  до фандинга (за 5 мин): {pre.mean()*1e4:+.3f} bps/мин | "
          f"после (5 мин): {post.mean()*1e4:+.3f} bps/мин")

print("\n(скрин завершён; отбор — не вердикт)")
