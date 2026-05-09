#!/usr/bin/env python3
"""Run replicated gem5 sweeps for the real-sized FHE RNS microkernel."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


METRICS = [
    "simSeconds",
    "simTicks",
    "hostSeconds",
    "simInsts",
    "system.cpu.numCycles",
    "system.cpu.cpi",
    "system.cpu.ipc",
    "system.cpu.dcache.overallAccesses::total",
    "system.cpu.dcache.overallMisses::total",
    "system.cpu.dcache.overallMissRate::total",
    "system.cpu.icache.overallAccesses::total",
    "system.cpu.icache.overallMisses::total",
    "system.cpu.icache.overallMissRate::total",
    "system.l2.overallAccesses::total",
    "system.l2.overallMisses::total",
    "system.l2.overallMissRate::total",
]

CONFIGS = {
    "nocache": {"caches": False, "l2": False},
    "l1": {"caches": True, "l2": False},
    "l1_l2": {"caches": True, "l2": True},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="data/fhe_rns_microkernel_sweep")
    parser.add_argument("--m5out-root", default="m5out/fhe_rns_microkernel_sweep")
    parser.add_argument("--configs", default="l1_l2", help="Comma-separated: nocache,l1,l1_l2")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2, help="Kernel repetitions inside each ROI run")
    parser.add_argument("--gem5-dir", default="tools/gem5")
    parser.add_argument("--openfhe-bin-dir", default=os.environ.get("OPENFHE_BIN_DIR", "/home/jzhao7/RAGPIR/openfhe_core/build/bin"))
    parser.add_argument("--centroids-file", default=os.environ.get("OPENFHE_CENTROIDS_FILE", ""))
    parser.add_argument("--encrypted-query", default=os.environ.get("OPENFHE_ENCRYPTED_QUERY", ""))
    parser.add_argument("--encrypted-norm", default=os.environ.get("OPENFHE_ENCRYPTED_NORM", ""))
    parser.add_argument("--extra-gem5-args", default=os.environ.get("GEM5_EXTRA_ARGS", ""))
    parser.add_argument("--skip-export", action="store_true", help="Reuse <work-dir>/distance_operands.bin")
    return parser


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("Running:", shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def resolve_gem5_env(gem5_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    candidates = [
        Path.home() / "miniconda3/envs/gem5env/lib",
        Path.home() / "anaconda3/envs/gem5env/lib",
    ]
    # Avoid the base conda lib directory here: its libstdc++ can be older than
    # the one used to build gem5. The gem5env path provides the needed libpython.
    existing = [str(path) for path in candidates if path.exists()]
    if existing:
        env["LD_LIBRARY_PATH"] = ":".join(existing + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    return env


def export_operands(args: argparse.Namespace, work_dir: Path) -> Path:
    operands = work_dir / "distance_operands.bin"
    if args.skip_export and operands.exists():
        return operands
    required = {
        "centroids file": args.centroids_file,
        "encrypted query": args.encrypted_query,
        "encrypted norm": args.encrypted_norm,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required inputs for export: {', '.join(missing)}")
    work_dir.mkdir(parents=True, exist_ok=True)
    exporter = Path(args.openfhe_bin_dir) / "openfhe_export_distance_operands"
    run([
        str(exporter),
        "--centroids-file",
        args.centroids_file,
        "--encrypted-query",
        args.encrypted_query,
        "--encrypted-norm",
        args.encrypted_norm,
        "--output-dir",
        str(work_dir),
    ])
    return operands


def first_stats_block(path: Path) -> dict[str, float]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("---------- Begin Simulation Statistics"):
            current = []
        elif line.startswith("---------- End Simulation Statistics"):
            if current is not None:
                blocks.append(current)
                current = None
        elif current is not None:
            current.append(line)
    if not blocks:
        raise ValueError(f"No stats blocks found in {path}")
    values: dict[str, float] = {}
    for line in blocks[0]:
        parts = line.split()
        if len(parts) >= 2 and parts[0] in METRICS:
            values[parts[0]] = float(parts[1])
    values["stats_blocks"] = float(len(blocks))
    return values


def run_trial(
    *,
    config: str,
    trial: int,
    args: argparse.Namespace,
    operands: Path,
    gem5_bin: Path,
    gem5_se: Path,
    kernel_bin: Path,
    outdir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    cfg = CONFIGS[config]
    kernel_output = Path(args.work_dir) / f"kernel_output_{config}_trial{trial}.json"
    cmd = [
        str(gem5_bin),
        f"--outdir={outdir}",
        str(gem5_se),
        f"--cmd={kernel_bin}",
        "--options="
        + " ".join(
            [
                "--operands",
                str(operands),
                "--output",
                str(kernel_output),
                "--repeats",
                str(args.repeats),
                "--gem5-roi",
                "1",
            ]
        ),
        "--cpu-type=AtomicSimpleCPU",
        "--mem-size=4GB",
    ]
    if cfg["caches"]:
        cmd.append("--caches")
    if cfg["l2"]:
        cmd.append("--l2cache")
    if args.extra_gem5_args:
        cmd.extend(shlex.split(args.extra_gem5_args))
    run(cmd, env=env)
    row: dict[str, Any] = {
        "config": config,
        "trial": trial,
        "repeats": args.repeats,
        "stats": str(outdir / "stats.txt"),
        "kernel_output": str(kernel_output),
    }
    row.update(first_stats_block(outdir / "stats.txt"))
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": rows, "groups": {}}
    configs = sorted({str(row["config"]) for row in rows})
    for config in configs:
        group_rows = [row for row in rows if row["config"] == config]
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            vals = [float(row[metric]) for row in group_rows if metric in row]
            if not vals:
                continue
            metrics[metric] = {
                "n": len(vals),
                "mean": statistics.fmean(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
            }
        summary["groups"][config] = metrics
    return summary


def write_outputs(work_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fieldnames = ["config", "trial", "repeats", "stats", "kernel_output"] + METRICS
    with (work_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# FHE RNS Microkernel Sweep Report", ""]
    for config, metrics in summary["groups"].items():
        lines.append(f"## {config}")
        for metric in [
            "simInsts",
            "system.cpu.numCycles",
            "system.cpu.cpi",
            "system.cpu.ipc",
            "system.cpu.dcache.overallMissRate::total",
            "system.l2.overallMissRate::total",
            "hostSeconds",
        ]:
            if metric not in metrics:
                continue
            stat = metrics[metric]
            lines.append(
                f"- `{metric}`: mean={stat['mean']:.6g}, stdev={stat['stdev']:.6g}, "
                f"min={stat['min']:.6g}, max={stat['max']:.6g}, n={stat['n']}"
            )
        lines.append("")
    (work_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    configs = [part.strip() for part in args.configs.split(",") if part.strip()]
    unknown = [cfg for cfg in configs if cfg not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configs: {', '.join(unknown)}")

    work_dir = Path(args.work_dir)
    m5out_root = Path(args.m5out_root)
    gem5_dir = Path(args.gem5_dir)
    gem5_bin = gem5_dir / "build/X86/gem5.opt"
    gem5_se = gem5_dir / "configs/deprecated/example/se.py"
    kernel_bin = Path(args.openfhe_bin_dir) / "fhe_rns_distance_kernel"

    operands = export_operands(args, work_dir)
    env = resolve_gem5_env(gem5_bin)

    rows: list[dict[str, Any]] = []
    for config in configs:
        for trial in range(1, args.trials + 1):
            outdir = m5out_root / f"{config}_r{args.repeats}_trial{trial}"
            if outdir.exists():
                subprocess.run(["rm", "-rf", str(outdir)], check=True)
            row = run_trial(
                config=config,
                trial=trial,
                args=args,
                operands=operands,
                gem5_bin=gem5_bin,
                gem5_se=gem5_se,
                kernel_bin=kernel_bin,
                outdir=outdir,
                env=env,
            )
            rows.append(row)
            write_outputs(work_dir, rows, summarize(rows))

    write_outputs(work_dir, rows, summarize(rows))
    print(f"Wrote {work_dir / 'summary.json'}")
    print(f"Wrote {work_dir / 'summary.csv'}")
    print(f"Wrote {work_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_fhe_rns_microkernel_sweep error: {exc}", file=sys.stderr)
        raise SystemExit(1)
