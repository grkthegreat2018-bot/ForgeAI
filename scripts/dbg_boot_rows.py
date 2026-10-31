import sqlite3, time
cur = sqlite3.connect("forge_evolve.db").cursor()
boot = time.time() - 1800  # last 30 min
rows = cur.execute(
    "SELECT ts, ok, stage, adapter FROM code_programs WHERE ts > ? ORDER BY id",
    (boot,)).fetchall()
print(f"last-30min rows: {len(rows)}")
for r in rows[-15:]:
    print(time.strftime("%H:%M:%S", time.localtime(r[0])),
          "ok=", r[1], "stage=", r[2], "ad=", r[3])
if rows:
    print(f"ok rate: {100*sum(r[1] for r in rows)/len(rows):.0f}%")
