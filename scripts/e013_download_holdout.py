# E013 — скачивание HOLDOUT-данных по альтам: 2026-01..2026-10 (1h)
# Запуск ПОСЛЕ замороженной регистрации E013_PREREGISTRATION.md (коммит e4668f1).
# 2026-10 = дневные архивы; 2026-10-28 биржей не опубликован (известный пропуск,
# как в E012). SHA256-манифест печатается.
import hashlib
import os
import subprocess

RAW = "/home/z/my-project/binfut-platform-restore/data/raw/klines1h"
ALTS = ["SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT"]
MON = ("https://data.binance.vision/data/futures/um/monthly/klines"
       "/{s}/1h/{s}-1h-2026-{m:02d}.zip")
DAY = ("https://data.binance.vision/data/futures/um/daily/klines"
       "/{s}/1h/{s}-1h-2026-10-{d:02d}.zip")

os.makedirs(RAW, exist_ok=True)
jobs = []
for s in ALTS:
    for m in range(1, 10):
        jobs.append((MON.format(s=s, m=m), f"{RAW}/{s}-1h-2026-{m:02d}.zip"))
    for d in range(1, 32):
        jobs.append((DAY.format(s=s, d=d), f"{RAW}/{s}-1h-2026-10-{d:02d}.zip"))


def fetch(job):
    url, dest = job
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return ("OK", dest, None)
    r = subprocess.run(["curl", "-fsS", "--max-time", "60", "-o", dest, url],
                       capture_output=True)
    if r.returncode != 0:
        if os.path.exists(dest):
            os.remove(dest)
        return ("MISS", url, None)
    return ("OK", dest, None)


from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as ex:
    out = list(ex.map(fetch, jobs))

ok = sorted(d for s, d, _ in out if s == "OK")
miss = sorted(u for s, u, _ in out if s == "MISS")
print(f"скачано: {len(ok)}; отсутствует: {len(miss)}")
for u in miss:
    print("  MISS:", u.split('/klines/')[1])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


print("SHA256-манифест (первые 3 месяца каждого символа):")
for s in ALTS:
    row = [sha256(f"{RAW}/{s}-1h-2026-{m:02d}.zip") for m in (1, 2, 3)]
    print(f"  {s}: " + " ".join(row))
