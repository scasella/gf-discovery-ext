"""Runner + swap watchdog for one launched job (PROCESS.md §7.3–7.4). Started detached by tools/launch.py; not run by hand.

Runs each phase of one script in sequence (a chained verdict is just the last phase), each in its own process group, and:
  - kills the phase's process group if machine-wide swap exceeds --swap-limit GB (default 12; other projects count too);
  - writes the status file (logs/<run>.status.json) at start, every 60 s (heartbeat), at each phase end and at the end;
  - records every phase's exit code, peak swap, peak process-group RSS and whether the watchdog killed it;
  - records the docs/HASHES.txt lines the job appended.
Chained phases run whatever the previous phase's exit code was (the verdict phase must handle missing inputs itself),
unless --stop-on-fail is given. A heartbeat older than 3 min with no "end" means the runner died: CRASHED. A runner error is recorded (exit 99).
"""
import argparse, json, os, re, signal, subprocess, sys, time
from pathlib import Path

HEARTBEAT_S, POLL_S = 60, 10


def swap_gb():
    m = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout)
    return float(m.group(1)) / 1024 if m else float("nan")


def pgroup_rss_gb(pgid):
    out = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True).stdout
    return sum(int(r) for g, r in (l.split() for l in out.splitlines() if l.strip()) if int(g) == pgid) / 1024 ** 2


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--phase", action="append", required=True)
    ap.add_argument("--extra", default="", help="arguments appended to every phase command (one string)")
    ap.add_argument("--swap-limit", type=float, default=12.0)
    ap.add_argument("--heartbeat-s", type=int, default=HEARTBEAT_S)
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--meta", default="{}", help="JSON merged into the status file (session, expected_s, shas, env)")
    a = ap.parse_args()
    ext, status_p = Path(a.ext), Path(a.status)
    hashes = ext / "docs/HASHES.txt"
    n_hash0 = len(hashes.read_text().splitlines()) if hashes.exists() else 0
    st = dict(run=a.run, runner_pid=os.getpid(), start=now(), heartbeat=now(), end=None, exit_code=None,
              swap_limit_gb=a.swap_limit, peak_swap_gb=0.0, phases=[], hashes_appended=[], **json.loads(a.meta))

    def write():
        st["heartbeat"] = now()
        tmp = status_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, status_p)

    write()
    logf = open(a.log, "a")
    try:
        run_phases(a, ext, st, write, logf)
    except BaseException as ex:                                   # a runner failure is recorded, never silent
        st["runner_error"] = repr(ex)
        logf.write(f"{now()} RUNNER ERROR {ex!r}\n"); logf.flush()
    lines = hashes.read_text().splitlines() if hashes.exists() else []
    st["hashes_appended"] = lines[n_hash0:]
    codes = [r["exit_code"] for r in st["phases"]]
    st["exit_code"] = 99 if "runner_error" in st else next((c for c in codes if c != 0), 0)
    st["end"] = now()
    write()
    logf.write(f"===== {now()} run {a.run} end; exit {st['exit_code']}; peak swap {st['peak_swap_gb']:.2f} GB\n"); logf.close()


def run_phases(a, ext, st, write, logf):
    for ph in a.phase:
        cmd = [a.python, a.script, "--phase", ph] + (a.extra.split() if a.extra else [])
        logf.write(f"\n===== {now()} phase {ph}: {' '.join(cmd)}\n"); logf.flush()
        p = subprocess.Popen(cmd, cwd=ext, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        rec = dict(phase=ph, cmd=cmd, pid=p.pid, pgid=os.getpgid(p.pid), start=now(), end=None, exit_code=None,
                   peak_swap_gb=0.0, peak_rss_gb=0.0, killed=None)
        st["phases"].append(rec); write()
        last = time.time()
        while p.poll() is None:
            s = swap_gb()
            rec["peak_swap_gb"] = max(rec["peak_swap_gb"], s); st["peak_swap_gb"] = max(st["peak_swap_gb"], s)
            try:
                rec["peak_rss_gb"] = max(rec["peak_rss_gb"], pgroup_rss_gb(rec["pgid"]))
            except Exception:
                pass
            if s > a.swap_limit:
                rec["killed"] = f"{now()} swap {s:.2f} GB > {a.swap_limit} GB"
                logf.write(f"{now()} WATCHDOG: {rec['killed']}: killing process group {rec['pgid']}\n"); logf.flush()
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(rec["pgid"], sig)
                    except (ProcessLookupError, PermissionError):   # group gone (or only a zombie left): done
                        break
                    try:
                        p.wait(timeout=10); break
                    except subprocess.TimeoutExpired:
                        pass
                p.wait(); break
            if time.time() - last >= a.heartbeat_s:
                last = time.time(); write()
            time.sleep(min(POLL_S, a.heartbeat_s))
        rec["exit_code"], rec["end"] = p.returncode, now()
        logf.write(f"===== {now()} phase {ph} exit {p.returncode}; peak swap {rec['peak_swap_gb']:.2f} GB; "
                   f"peak RSS {rec['peak_rss_gb']:.2f} GB{'; KILLED' if rec['killed'] else ''}\n"); logf.flush()
        write()
        if p.returncode != 0 and a.stop_on_fail:
            break


if __name__ == "__main__":
    sys.exit(main())
