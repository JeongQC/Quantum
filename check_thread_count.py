#!/usr/bin/env python
"""Check current process thread count. Exits 1 if approaching macOS ulimit.

Used by ``rerun_missing_cs.sh`` (and similar launchers) BEFORE starting a
long QPU sweep, so we abort early if another process is already consuming
most of the per-user thread/process budget.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys


def main():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        print(f"ulimit -u: soft={soft}, hard={hard}")
        user = os.getenv("USER", "")
        if not user:
            print("USER env var unset; cannot count threads. Treating as OK.")
            return 0
        # `ps -u <user> -M` lists all threads for the user (one row per
        # thread). The header line is one row; subtract it.
        result = subprocess.run(
            ["ps", "-u", user, "-M"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"ps failed (rc={result.returncode}); treating as OK")
            print(f"stderr: {result.stderr.strip()[:200]}")
            return 0
        n_threads = max(0, len(result.stdout.splitlines()) - 1)
        print(f"current threads for user {user}: {n_threads}")
        if soft <= 0:
            print("ulimit soft <= 0; cannot compute utilization. Treating as OK.")
            return 0
        utilization = n_threads / float(soft)
        print(f"utilization: {utilization:.1%}")
        if utilization > 0.8:
            print("WARNING: thread utilization >80%, abort and clean up "
                  "before launching")
            return 1
        print("OK to launch")
        return 0
    except Exception as e:
        print(f"check failed: {e.__class__.__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
