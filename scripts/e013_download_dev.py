# E013 — скачивание DEV-данных: 8 альтов + BTC/ETH, 1h, 2021-01..2025-12
# Источник: data.binance.vision (публичные архивы UM futures).
# Holdout 2026 НЕ качается этим скриптом (отдельно, после заморозки регистрации).
import os
import subprocess
import sys

RAW = "/home/z/my-project/binfut-platform-restore/data/raw/klines1h"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT"]
YEARS = range(2021, 2026)          # dev: 2021-01 .. 2025-12
BASE = ("https://data.binance.vision/data/futures/um/monthly/klines"
        "/{sym}/1h/{sym}-1h-{y}-{m:02d}.zip")

os.makedirs(RAW, exist_ok=True)
jobs, missing = [], []
for sym in SYMS:
    for y in YEARS:
        for m in range(1, 13):
            dest = f"{RAW}/{sym}-1h-{y}-{m:02d}.zip"
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue
            url = BASE.format(sym=sym, y=y, m=m)
            jobs.append((url, dest))

print(f"к скачиванию: {len(jobs)} файлов")


def fetch(job):
    url, dest = job
    r = subprocess.run(["curl", "-fsS", "--max-time", "60",
                        "-o", dest, url],
                       capture_output=True)
    if r.returncode != 0:
        # 404 = контракт ещё не существовал (начало 2021 у DOT/SOL и т.п.)
        if os.path.exists(dest):
            os.remove(dest)
        return ("MISS", url)
    return ("OK", dest)


from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as ex:
    out = list(ex.map(fetch, jobs))

ok = [d for s, d in out if s == "OK"]
missing = sorted(url.split('/klines/')[1] for s, url in out if s == "MISS")
print(f"скачано: {len(ok)}; отсутствует на бирже: {len(missing)}")
if missing:
    print("первые пропуски (ожидаемо для ранних месяцев новых листингов):")
    for u in missing[:12]:
        print("  ", u)
sizes = sum(os.path.getsize(f"{RAW}/{f}") for f in os.listdir(RAW)
           if f.endswith('.zip'))
print(f"итого в кэше: {len([f for f in os.listdir(RAW) if f.endswith('.zip')])} zip, "
      f"{sizes/1e6:.1f} МБ")
