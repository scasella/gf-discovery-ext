"""Launch one job detached under caffeinate with the swap watchdog and a chained verdict (PROCESS.md §7.1, §7.3).

  .venv/bin/python tools/launch.py --run e73 --script scripts/e73_coverage_admission.py \
      --phase calibrate --phase verdict --expected-s 3600 --declared-out-gb 0.1 \
      --require-hashed NOTE_coverage_admission_L12.md scripts/e73_coverage_admission.py

Preflight (refuses to launch on any failure): the status file is new (a run launches once); every --require-hashed file's
sha is in docs/HASHES.txt; machine swap < --max-swap GB; free disk >= 8 GB and folder + declared output <= 17 GB; no other
gf-discovery-ext e6x/e7x job is running; no python process above 4 GB RSS (other projects' jobs are named, never touched;
--allow-big-other overrides and is recorded). Then it starts tools/watchdog.py as the runner, detached (own session) under
`caffeinate -i`, and returns. The runner runs the phases in order, writes logs/<run>.status.json (60 s heartbeat, per-phase
exit code, peak swap) and kills a phase's process group above --swap-limit GB.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

EXT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def swap_gb():
    m = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout)
    return float(m.group(1)) / 1024 if m else float("nan")


def folder_gb(p):
    out = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True).stdout.split()
    return int(out[0]) / 1024 ** 2


def python_procs():
    """Python processes (executable basename starts with "python"), excluding this launcher and its ancestors, whose
    shell command lines name the script being launched."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss=,command="], capture_output=True, text=True).stdout
    procs = {}
    for l in out.splitlines():
        parts = l.split(None, 3)
        if len(parts) == 4:
            procs[int(parts[0])] = (int(parts[1]), int(parts[2]), parts[3])
    anc, p = set(), os.getpid()
    while p in procs and p not in anc and p > 1:
        anc.add(p); p = procs[p][0]
    rows = []
    for pid, (_, rss, cmd) in procs.items():
        exe = os.path.basename(cmd.split()[0]) if cmd.split() else ""
        if pid not in anc and exe.lower().startswith("python"):
            rows.append((pid, rss / 1024 ** 2, cmd))
    return rows


def preflight(a, status_p):
    fails, notes = [], []
    if status_p.exists():
        fails.append(f"{status_p} exists: a run launches once (pick a new --run name only for a different job)")
    if not (EXT / a.script).exists():
        fails.append(f"{a.script} missing")
    hashes = (EXT / "docs/HASHES.txt").read_text()
    for rel in a.require_hashed:
        if not (EXT / rel).exists() or sha(EXT / rel) not in hashes:
            fails.append(f"{rel} is not hashed in docs/HASHES.txt (or changed after hashing)")
    s = swap_gb()
    if not s < a.max_swap:
        fails.append(f"swap {s:.2f} GB >= {a.max_swap} GB")
    free = shutil.disk_usage(EXT).free / 1024 ** 3
    fold = folder_gb(EXT)
    if free < 8:
        fails.append(f"free disk {free:.1f} GB < 8 GB")
    if fold + a.declared_out_gb > 17:
        fails.append(f"folder {fold:.1f} GB + declared {a.declared_out_gb} GB > 17 GB")
    for pid, rss, cmd in python_procs():
        if re.search(r"scripts/e[67]\d", cmd):
            fails.append(f"another gf-discovery-ext job is running: pid {pid} ({cmd[:120]})")
        if rss > 4:
            msg = f"python pid {pid} at {rss:.1f} GB RSS: {cmd[:160]}"
            (notes if a.allow_big_other else fails).append(msg)
    return fails, notes, dict(swap_gb=round(s, 2), free_disk_gb=round(free, 1), folder_gb=round(fold, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--script", required=True, help="path relative to the project root")
    ap.add_argument("--phase", action="append", required=True, help="repeat; phases run in this order (verdict last)")
    ap.add_argument("--extra", default="", help="arguments appended to every phase command")
    ap.add_argument("--expected-s", type=int, default=3600)
    ap.add_argument("--declared-out-gb", type=float, default=0.1)
    ap.add_argument("--require-hashed", nargs="*", default=[])
    ap.add_argument("--max-swap", type=float, default=8.0)
    ap.add_argument("--swap-limit", type=float, default=12.0)
    ap.add_argument("--heartbeat-s", type=int, default=60)
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--allow-big-other", action="store_true")
    ap.add_argument("--session", default=os.environ.get("CLAUDE_SESSION_ID", ""))
    ap.add_argument("--logs", default=str(EXT / "logs"))
    ap.add_argument("--dry-run", action="store_true", help="preflight only")
    a = ap.parse_args()
    logs = Path(a.logs); logs.mkdir(exist_ok=True)
    status_p, log_p = logs / f"{a.run}.status.json", logs / f"{a.run}.log"
    fails, notes, pre = preflight(a, status_p)
    for n in notes:
        print("NOTE:", n)
    if fails:
        print("PREFLIGHT REFUSED:"); [print("  -", f) for f in fails]
        return 2
    print("preflight ok", pre)
    if a.dry_run:
        return 0
    py = str(EXT / ".venv/bin/python")
    env = dict(os.environ, HF_HOME=str(EXT / "hf_home"), HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    meta = dict(session=a.session, expected_s=a.expected_s, script=a.script, script_sha256=sha(EXT / a.script),
                required_hashed={r: sha(EXT / r) for r in a.require_hashed}, preflight=pre, preflight_notes=notes,
                env=dict(HF_HOME=env["HF_HOME"], HF_HUB_OFFLINE="1"), launched=time.strftime("%Y-%m-%dT%H:%M:%S"))
    cmd = ["caffeinate", "-i", py, str(EXT / "tools/watchdog.py"), "--ext", str(EXT), "--run", a.run, "--status", str(status_p),
           "--log", str(log_p), "--python", py, "--script", a.script, "--swap-limit", str(a.swap_limit),
           "--heartbeat-s", str(a.heartbeat_s), "--meta", json.dumps(meta)]
    for ph in a.phase:
        cmd += ["--phase", ph]
    if a.extra:
        cmd += ["--extra", a.extra]
    if a.stop_on_fail:
        cmd += ["--stop-on-fail"]
    p = subprocess.Popen(cmd, cwd=EXT, env=env, stdout=subprocess.DEVNULL, stderr=open(logs / f"{a.run}.runner.err", "w"),
                         stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        if status_p.exists():
            break
        time.sleep(0.1)
    print(f"launched {a.run}: runner pid {p.pid} (caffeinate), phases {a.phase}; status {status_p}; log {log_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
