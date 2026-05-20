#!/usr/bin/env python3
"""
Tiny helpers for controlling Cargo parallelism.

Batch runs often execute many projects in parallel. If each `cargo build/check`
uses the default job count (≈ CPU cores), overall throughput can get worse due
to oversubscription and memory pressure. This module injects `-j <N>` into
Cargo commands based on environment variables.
"""

from __future__ import annotations

import os
from typing import List, Optional


def _read_int_env(keys: List[str]) -> Optional[int]:
    for k in keys:
        v = os.environ.get(k, "").strip()
        if not v:
            continue
        try:
            n = int(v)
        except Exception:
            continue
        if n > 0:
            return n
    return None


def get_cargo_jobs() -> Optional[int]:
    """
    Resolve desired cargo job count.

    Priority:
    - C2R_CARGO_JOBS (this repo)
    - CARGO_JOBS (generic)
    - CARGO_BUILD_JOBS (some environments)
    """
    return _read_int_env(["C2R_CARGO_JOBS", "CARGO_JOBS", "CARGO_BUILD_JOBS"])


def with_cargo_jobs(cmd: List[str]) -> List[str]:
    """
    If cmd is a cargo subcommand, inject `-j <N>` right after the subcommand
    unless already present.
    """
    jobs = get_cargo_jobs()
    if not jobs:
        return cmd
    if not cmd or cmd[0] != "cargo":
        return cmd
    if "-j" in cmd or "--jobs" in cmd:
        return cmd
    if len(cmd) < 2:
        return cmd
    sub = cmd[1]
    if sub not in ("build", "check", "test", "run"):
        return cmd
    return cmd[:2] + ["-j", str(jobs)] + cmd[2:]

