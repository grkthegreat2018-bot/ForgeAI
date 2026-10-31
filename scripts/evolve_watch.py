"""Live progress board for a code-evolution campaign.

Watches the shared evolution db + campaign log and redraws a compact
dashboard every --interval seconds: active runs (rows/evals/dup rate/
throughput), the all-time best score per target, and recent events
(NEW BEST / round / quiet-streak lines from the log).

Run in a second terminal:
    python scripts/evolve_watch.py --log evolve_campaign3.log
    python scripts/evolve_watch.py --db other.db --interval 5
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _active_runs(conn: sqlite3.Connection, limit: int = 8):
    """Latest run per target, with row counts + stage breakdown."""
    # newest run per target only — older campaigns' runs still show in
    # a raw MAX(id) ordering and read as "active" when they are dead.
    runs = conn.execute(
        "SELECT run_id, target, COUNT(*), "
        "SUM(CASE WHEN stage='done' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN stage='dup' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN stage IN ('syntax','missing','test','timeout',"
        "'spawn') THEN 1 ELSE 0 END), "
        "MAX(ts) FROM code_programs GROUP BY run_id, target "
        "ORDER BY MAX(id) DESC").fetchall()
    seen: set[str] = set()
    out = []
    for r in runs:
        if r[1] in seen:
            continue
        seen.add(r[1])
        out.append(r)
        if len(out) >= limit:
            break
    return out


def _best_scores(conn: sqlite3.Connection):
    return dict(conn.execute(
        "SELECT target, MAX(score) FROM code_programs "
        "WHERE ok=1 GROUP BY target").fetchall())


def _run_best(conn: sqlite3.Connection, run_id: str):
    r = conn.execute(
        "SELECT MAX(score) FROM code_programs WHERE run_id=? AND ok=1",
        (run_id,)).fetchone()
    return r[0] if r and r[0] is not None else float("nan")


def _log_events(path: Path, n: int = 10):
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    ev = [l.strip() for l in lines
          if re.search(r"NEW BEST|CAMPAIGN ROUND|quiet|>>>|CRASH|stagnation",
                       l)]
    return ev[-n:]


def _status_view(path: Path) -> str:
    """Current in-flight jobs + proposal counters from the run's
    status JSON (written by CodeEvolve._write_status ~every 2s)."""
    if not path or not path.exists():
        return ""
    try:
        import json
        st = json.loads(path.read_text(encoding="utf-8")
                        .splitlines()[-1])
    except Exception:
        return ""
    age = time.time() - float(st.get("ts", 0))
    lines = [f"status ({path.name}, {age:.0f}s old): "
             f"evals={st.get('evals')} promos={st.get('promos')} "
             f"dry={st.get('dry')} proposals={st.get('proposals')}"]
    inf = st.get("inflight") or []
    for j in inf[:12]:
        el = j.get("elapsed_s")
        over = j.get("over_s", 0)
        warn = "  <-- OVER deadline" if isinstance(over, (int, float)) \
            and over > 0 else ""
        lines.append(f"    in-flight: w{j.get('worker', '?')} "
                     f"{j.get('tag', '?')[:10]} "
                     f"{el if el is not None else '?'}s{warn}")
    if len(inf) > 12:
        lines.append(f"    ... +{len(inf) - 12} more in-flight")
    return "\n".join(lines)


def _render(conn, log: Path, hist: dict, interval: float,
            status: Path | None = None) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    now = time.strftime("%H:%M:%S")
    print(f"{'=' * 72}\n  FORGEAI CODE EVOLUTION — live board   {now}"
          f"   (refresh {interval:g}s, Ctrl+C to exit)\n{'=' * 72}")

    sv = _status_view(status) if status else ""
    if sv:
        print(f"\n{sv}")

    runs = _active_runs(conn)
    print(f"\n{'run':16s} {'target':18s} {'rows':>5} {'done':>5} "
          f"{'dup':>5} {'fail':>5} {'rows/m':>6} {'done/m':>6} "
          f"{'run-best':>9}")
    prev = hist.get("runs", {})
    for rid, tgt, rows, done, dup, fail, ts in runs:
        pr = prev.get(rid)
        rate = dm = 0.0
        if pr and interval > 0:
            rate = (rows - pr[0]) / (interval / 60)
            dm = (done - pr[1]) / (interval / 60)
        best = _run_best(conn, rid)
        print(f"{rid[:16]:16s} {tgt:18s} {rows:>5} {done:>5} {dup:>5} "
              f"{fail:>5} {rate:>6.1f} {dm:>6.1f} {best:>9.4f}")
        prev[rid] = (rows, done)
    hist["runs"] = prev

    print(f"\nall-time best score per target (across every run):")
    best = _best_scores(conn)
    for t in sorted(best, key=best.get):
        print(f"    {t:20s} {best[t]:+.5f}")

    ev = _log_events(log)
    if ev:
        print(f"\nrecent events:")
        for l in ev:
            print(f"    {l[:110]}")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="forge_evolve.db")
    ap.add_argument("--log", default="evolve_campaign3.log")
    ap.add_argument("--status", default=None,
                    help="status JSON written by --status-file — shows "
                         "in-flight jobs (the 'what is it doing NOW' "
                         "view) on the board")
    ap.add_argument("--interval", type=float, default=15.0)
    args = ap.parse_args()
    hist: dict = {}
    while True:
        try:
            conn = sqlite3.connect(
                f"file:{args.db}?mode=ro", uri=True, timeout=5)
            _render(conn, Path(args.log), hist, args.interval,
                    Path(args.status) if args.status else None)
            conn.close()
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            print(f"[watch] {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
