"""Smoke test for tools/launch.py + tools/watchdog.py (PROCESS.md §8 item 0). No project data, no HASHES write.

Writes a stub script and a logs dir under a temp dir, then checks:
  A  phases ok (exit 0), fail (exit 7) and a chained verdict (exit 0) run in order; the verdict runs after the failure;
     the status file records exit codes [0, 7, 0], overall exit 7, an end time and >= 2 heartbeats (heartbeat 2 s here);
  B  --swap-limit 0 makes the watchdog kill the phase's process group (killed recorded, exit code < 0);
  C  preflight refuses an unhashed --require-hashed file and an existing status file (a run launches once).
Usage: .venv/bin/python docs/launch_smoke.py [--keep DIR]
"""
import json, subprocess, sys, tempfile, time
from pathlib import Path

EXT = Path(__file__).resolve().parents[1]
PY = str(EXT / ".venv/bin/python")
STUB = '''import sys, time, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--phase"); a, _ = ap.parse_known_args()
print("stub phase", a.phase, flush=True)
time.sleep({"ok": 5, "fail": 1, "verdict": 1, "long": 60}[a.phase])
sys.exit(7 if a.phase == "fail" else 0)
'''


def launch(tmp, run, phases, *extra):
    cmd = [PY, str(EXT / "tools/launch.py"), "--run", run, "--script", str(tmp / "stub.py"), "--logs", str(tmp / "logs"),
           "--expected-s", "30", "--declared-out-gb", "0", "--heartbeat-s", "2", "--allow-big-other", *extra]
    for ph in phases:
        cmd += ["--phase", ph]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip()); return r


def wait_end(p, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.exists():
            st = json.loads(p.read_text())
            if st.get("end"):
                return st
        time.sleep(1)
    raise AssertionError(f"{p} never ended")


def main():
    tmp = Path(sys.argv[sys.argv.index("--keep") + 1]) if "--keep" in sys.argv else Path(tempfile.mkdtemp(prefix="launch_smoke_"))
    tmp.mkdir(parents=True, exist_ok=True); (tmp / "stub.py").write_text(STUB)
    hb = []
    # A
    r = launch(tmp, "smokeA", ["ok", "fail", "verdict"]); assert r.returncode == 0, r.stderr
    p = tmp / "logs/smokeA.status.json"
    t0 = time.time()
    while time.time() - t0 < 60:
        if p.exists():
            st = json.loads(p.read_text())
            if st["heartbeat"] not in hb:
                hb.append(st["heartbeat"])
            if st.get("end"):
                break
        time.sleep(0.5)
    st = wait_end(p)
    codes = [r_["exit_code"] for r_ in st["phases"]]
    assert [r_["phase"] for r_ in st["phases"]] == ["ok", "fail", "verdict"], st["phases"]
    assert codes == [0, 7, 0] and st["exit_code"] == 7, codes
    assert len(hb) >= 3, f"heartbeats seen {hb}"
    assert "stub phase verdict" in (tmp / "logs/smokeA.log").read_text()
    print("A ok: exit codes", codes, "overall", st["exit_code"], "heartbeats", len(hb))
    # B
    r = launch(tmp, "smokeB", ["long"], "--swap-limit", "0", "--max-swap", "100"); assert r.returncode == 0, r.stderr
    st = wait_end(tmp / "logs/smokeB.status.json")
    ph = st["phases"][0]
    assert ph["killed"] and ph["exit_code"] < 0 and st["exit_code"] == ph["exit_code"], ph
    print("B ok: killed", ph["killed"], "exit", ph["exit_code"])
    # C
    (tmp / "unhashed.txt").write_text(f"not hashed {time.time()}")
    r = launch(tmp, "smokeC", ["ok"], "--require-hashed", str(tmp / "unhashed.txt"))
    assert r.returncode == 2 and "not hashed" in r.stdout and not (tmp / "logs/smokeC.status.json").exists(), r.stdout
    r = launch(tmp, "smokeA", ["ok"])
    assert r.returncode == 2 and "launches once" in r.stdout, r.stdout
    print("C ok: preflight refusals")
    print("LAUNCH SMOKE PASS", tmp)


if __name__ == "__main__":
    main()
