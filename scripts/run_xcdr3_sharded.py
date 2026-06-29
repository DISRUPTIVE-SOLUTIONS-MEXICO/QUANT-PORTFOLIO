from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "research_artifacts"
DEFAULT_PARTIALS_DIR = DEFAULT_OUT_DIR / "xcdr_partials"


def _stream_process(prefix: str, proc: subprocess.Popen[str]) -> int:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{prefix}] {line.rstrip()}", flush=True)
    return int(proc.wait())


def _run_one(args: list[str], env: dict[str, str], prefix: str) -> int:
    proc = subprocess.Popen(
        args,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return _stream_process(prefix, proc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run XCDR v3 research in bounded shards, then merge. This keeps the "
            "same research script and gates while making cost-zero runners less brittle."
        )
    )
    parser.add_argument("--shards", type=int, default=int(os.getenv("QPK_XCDR3_RUNNER_SHARDS", "6")))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("QPK_XCDR3_RUNNER_CONCURRENCY", "2")))
    parser.add_argument("--out-dir", type=Path, default=Path(os.getenv("QPK_XCDR3_OUT_DIR", str(DEFAULT_OUT_DIR))))
    parser.add_argument(
        "--partials-dir",
        type=Path,
        default=Path(os.getenv("QPK_XCDR3_PARTIALS_DIR", str(DEFAULT_PARTIALS_DIR))),
    )
    parser.add_argument("--clean-partials", action="store_true", default=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    shards = max(1, int(args.shards))
    concurrency = max(1, min(int(args.concurrency), shards))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.partials_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_partials:
        for path in args.partials_dir.glob("xcdr_v3_shard_*.csv"):
            path.unlink(missing_ok=True)

    base_env = os.environ.copy()
    base_env["QPK_XCDR3_OUT_DIR"] = str(args.out_dir)
    base_env["QPK_XCDR3_PARTIALS_DIR"] = str(args.partials_dir)
    base_env.setdefault("QPK_XCDR3_MAX_WINDOWS", "18")
    base_env.setdefault("QPK_XCDR3_UNIVERSE_LIMIT", "90")
    base_env.setdefault("QPK_XCDR3_BOOTSTRAP_N", "300")
    base_env.setdefault("QPK_XCDR3_PSO_PARTICLES", "18")
    base_env.setdefault("QPK_XCDR3_PSO_ITERATIONS", "18")
    # Each shard is a separate process; keep internal thread fanout modest.
    base_env.setdefault("QPK_XCDR3_WORKERS", "1")

    running: list[tuple[int, subprocess.Popen[str]]] = []
    failures: list[tuple[int, int]] = []
    next_shard = 0
    while next_shard < shards or running:
        while next_shard < shards and len(running) < concurrency:
            cmd = [
                args.python,
                "run_xcdr_v3_parallel_research.py",
                "--shard-index",
                str(next_shard),
                "--shard-count",
                str(shards),
                "--partials-dir",
                str(args.partials_dir),
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=base_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            print(f"[runner] started shard {next_shard}/{shards} pid={proc.pid}", flush=True)
            running.append((next_shard, proc))
            next_shard += 1

        still_running: list[tuple[int, subprocess.Popen[str]]] = []
        for shard, proc in running:
            if proc.poll() is None:
                still_running.append((shard, proc))
                continue
            if proc.stdout is not None:
                for line in proc.stdout.readlines():
                    print(f"[shard {shard}] {line.rstrip()}", flush=True)
            code = int(proc.returncode or 0)
            if code != 0:
                failures.append((shard, code))
                print(f"[runner] shard {shard} failed with exit code {code}", flush=True)
            else:
                print(f"[runner] shard {shard} completed", flush=True)
        running = still_running
        if failures:
            for _, proc in running:
                proc.terminate()
            return 1
        if running:
            time.sleep(2.0)

    merge_env = base_env.copy()
    merge_env["QPK_XCDR3_MERGE_PARTIALS"] = "1"
    merge_cmd = [
        args.python,
        "run_xcdr_v3_parallel_research.py",
        "--merge-partials",
        "--partials-dir",
        str(args.partials_dir),
    ]
    print("[runner] merging shard partials", flush=True)
    return _run_one(merge_cmd, merge_env, "merge")


if __name__ == "__main__":
    raise SystemExit(main())
