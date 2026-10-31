"""Proposal funnel over time: parse quality by era + adapter association."""
import sqlite3
import time

db = sqlite3.connect("forge_evolve.db")
cur = db.cursor()

# quality by time bucket: ok rate, code starts, adapter
rows = cur.execute(
    "SELECT ts, ok, score, adapter, code, target, run_id FROM code_programs "
    "ORDER BY id").fetchall()
print(f"total programs: {len(rows)}")
buckets = {}
for ts, ok, score, adapter, code, target, run_id in rows:
    hr = time.strftime("%d %H:%M", time.localtime(ts))
    b = buckets.setdefault(hr, [0, 0, 0.0])
    b[0] += 1
    b[1] += ok
    b[2] += score
for hr in sorted(buckets):
    n, ok, s = buckets[hr]
    print(f"{hr}  n={n:5d}  ok={ok:4d} ({100*ok/n:4.1f}%)  avg_score={s/n:12.2f}")

# last GOOD row (ok=1)
print("\n--- last 3 ok=1 rows ---")
for r in cur.execute(
        "SELECT ts, target, score, substr(code,1,200), adapter FROM code_programs "
        "WHERE ok=1 ORDER BY id DESC LIMIT 3"):
    print(time.strftime("%d %H:%M", time.localtime(r[0])), r[1], r[2],
          f"adapter={r[4]}", "\n  code:", r[3][:180].replace("\n", "\\n"))

# does babble correlate with adapter?
print("\n--- ok-rate by adapter (last 2000 rows) ---")
for r in cur.execute(
        "SELECT adapter, COUNT(*), SUM(ok) FROM (SELECT adapter, ok FROM "
        "code_programs ORDER BY id DESC LIMIT 2000) GROUP BY adapter"):
    print(r)
db.close()
