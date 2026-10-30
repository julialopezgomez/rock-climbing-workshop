#!/usr/bin/env python3
"""Run one GPU experiment when ambient load permits; recheck busy GPUs in 30 min.

Only the launched experiment's process group is managed. Other workloads are
never signalled. Use this serially for large models whose temporary allocations
can exceed their steady-state memory use.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ._paths import CACHE


def sample_gpu():
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    utilization, used, free, total = map(float, result.strip().split(","))
    return dict(utilization_percent=utilization, used_mib=used, free_mib=free, total_mib=total)


def utc():
    return dt.datetime.now(dt.UTC)


def run(args):
    args.state.parent.mkdir(parents=True, exist_ok=True)
    history = args.state.with_suffix(".jsonl")
    deadline = dt.datetime.fromisoformat(args.deadline) if args.deadline else None

    def record(value):
        value = dict(timestamp_utc=utc().isoformat(), **value)
        args.state.write_text(json.dumps(value, indent=2))
        with history.open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    lock = None
    if not args.check:
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        lock = args.lock.open("a+")
        announced = False
        while True:
            if deadline is not None and utc() >= deadline:
                record(dict(state="deadline_before_gpu_lock"))
                return 75
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not announced:
                    record(dict(state="waiting_for_owned_gpu_stage"))
                    announced = True
                time.sleep(5)
    while True:
        if deadline is not None and utc() >= deadline:
            record(dict(state="deadline_before_gpu_launch"))
            return 75
        samples = []
        for i in range(3):
            samples.append(sample_gpu())
            if i < 2:
                time.sleep(1)
        average = sum(s["utilization_percent"] for s in samples) / len(samples)
        free = min(s["free_mib"] for s in samples)
        allowed = average < args.busy_util and free >= args.peak_mib + args.reserve_mib
        status = dict(
            allowed=allowed,
            gpu=samples[-1],
            average_utilization_percent=average,
            required_free_mib=args.peak_mib + args.reserve_mib,
        )
        if args.check:
            record(dict(state="checked", **status))
            return 0 if allowed else 75
        if allowed:
            break
        if deadline is not None and utc() >= deadline:
            record(dict(state="deadline_reached_while_busy", **status))
            return 75
        next_check = utc() + dt.timedelta(seconds=args.poll_seconds)
        record(
            dict(state="waiting_for_ambient_gpu", next_check_utc=next_check.isoformat(), **status)
        )
        # Short sleeps keep termination responsive while maintaining the user's
        # requested 30-minute sampling interval.
        while utc() < next_check:
            if deadline is not None and utc() >= deadline:
                record(dict(state="deadline_reached_while_busy", **status))
                return 75
            time.sleep(min(30, max(0.1, (next_check - utc()).total_seconds())))
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Supply an experiment after --, or use --check")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        record(dict(state="running_owned_experiment", pid=process.pid, command=command, **status))

        def stop_owned(signum, frame):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, stop_owned)
        signal.signal(signal.SIGINT, stop_owned)
        returncode = process.wait()
    record(dict(state="finished", returncode=returncode, log=str(args.log), gpu=sample_gpu()))
    return returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(os.environ.get("RCR_GPU_LOCK", CACHE / "gpu.lock")),
    )
    parser.add_argument("--log", type=Path, default=Path("/tmp/perception-gpu-job.log"))
    parser.add_argument("--peak-mib", type=int, default=18000)
    parser.add_argument("--reserve-mib", type=int, default=3000)
    parser.add_argument("--busy-util", type=float, default=60)
    parser.add_argument("--poll-seconds", type=int, default=1800)
    parser.add_argument("--deadline")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    sys.exit(run(parser.parse_args()))
