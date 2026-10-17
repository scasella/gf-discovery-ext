"""Check docs/HASHES.txt against the files in this checkout.

Added for the public release; not itself in the ledger. Each ledger path gets one status:
  OK            the file matches the latest ledger entry for that path
  APPENDED      the file matches a "final sha256" that a ledger note records for it
                (e63_occurrence_v2.json was appended to after its freeze-time hash)
  NOT RELEASED  the file is listed in RELEASE_EXCLUDED.txt (too large; regenerable)
  ABSENT        the file is not here and not listed (deleted upstream after use)
  MISMATCH      the file is here but matches no entry: this is a failure

Run from the repository root: python3 tools/verify_hashes.py
Exit status is 1 if any path is MISMATCH, else 0.
"""
import hashlib
import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    entries = OrderedDict()   # path -> list of shas, ledger order
    finals = {}               # path -> final sha from a note line
    for line in (ROOT / "docs/HASHES.txt").read_text().splitlines():
        m = re.match(r"([0-9a-f]{64})\s+(\S+)", line)
        if m:
            entries.setdefault(m.group(2), []).append(m.group(1))
            continue
        m = re.search(r"(results/\S+) was appended to .* final sha256 is ([0-9a-f]{64})", line)
        if m:
            finals[m.group(1)] = m.group(2)
    excluded = set()
    ex = ROOT / "RELEASE_EXCLUDED.txt"
    if ex.exists():
        excluded = {l.split("\t")[0] for l in ex.read_text().splitlines() if l and not l.startswith("#")}

    counts = {}
    for path, shas in entries.items():
        p = ROOT / path
        if not p.exists():
            status = "NOT RELEASED" if path in excluded else "ABSENT"
        else:
            d = sha256(p)
            if d == shas[-1]:
                status = "OK"
            elif finals.get(path) == d:
                status = "APPENDED"
            else:
                status = "MISMATCH"
        counts[status] = counts.get(status, 0) + 1
        if status != "OK":
            print(f"{status:13s} {path}")
    n = sum(len(v) for v in entries.values())
    print(f"\n{n} ledger entries over {len(entries)} paths: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 1 if counts.get("MISMATCH") else 0


if __name__ == "__main__":
    sys.exit(main())
