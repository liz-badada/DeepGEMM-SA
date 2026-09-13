#!/usr/bin/env python3
"""CAKE evaluator for the H20 DeepSeek-V4-Flash MXFP4 MegaMoE kernel."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


FLASH_BATCHES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
PR383_US = {
    8: 273.1,
    16: 304.4,
    32: 302.0,
    64: 340.7,
    128: 414.4,
    256: 569.5,
    512: 922.0,
    1024: 1516.6,
    2048: 2735.1,
    4096: 5116.0,
    8192: 9749.0,
}
_ROW = re.compile(
    r"M=\s*(?P<m>\d+).*?fp8=\s*(?P<fp8>[0-9.]+)us.*?"
    r"mxfp4=\s*(?P<mxfp4>[0-9.]+)us"
)


def _run(command: list[str], log: Path) -> subprocess.CompletedProcess[str]:
    """Run one evaluator command and persist its combined output.

    :param command: Command and arguments to execute.
    :type command: list[str]
    :param log: Destination for combined standard output and error.
    :type log: pathlib.Path
    :return: Completed subprocess result.
    :rtype: subprocess.CompletedProcess[str]
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    env["NVIDIA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    env.setdefault("DG_BENCH_FLUSH_L2_BYTES", str(8_000_000_000))
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        timeout=7200,
    )
    log.write_text(proc.stdout, encoding="utf-8")
    return proc


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of an artifact.

    :param path: Artifact to hash.
    :type path: pathlib.Path
    :return: Lowercase hexadecimal digest.
    :rtype: str
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(name: str, passed: bool, artifact: Path, summary: str) -> dict[str, object]:
    """Build one evidence-backed evaluation gate.

    :param name: Stable gate name.
    :type name: str
    :param passed: Whether the gate passed.
    :type passed: bool
    :param artifact: Evidence artifact for the gate.
    :type artifact: pathlib.Path
    :param summary: Human-readable gate summary.
    :type summary: str
    :return: Serialized gate record.
    :rtype: dict[str, object]
    """
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "artifact": f"evidence/{artifact.name}",
        "artifact_sha256": _sha256(artifact),
        "summary": summary,
    }


def _wait_for_exclusive_gpus(log: Path) -> None:
    """Wait until all eight local H20 GPUs are free of material allocations.

    The direct-connect H20 hosts are shared with long-running inference jobs,
    outside CAKE's Slurm lease accounting. Refuse to begin an eight-rank run
    until every physical GPU is present and using at most the configured idle
    memory allowance.

    :param log: Destination for timestamped GPU availability observations.
    :type log: pathlib.Path
    :raises TimeoutError: If eight idle GPUs do not become available in time.
    """
    timeout_seconds = int(os.environ.get("DG_EVALUATOR_GPU_WAIT_SECONDS", "5400"))
    idle_memory_mib = int(os.environ.get("DG_EVALUATOR_IDLE_MEMORY_MIB", "1024"))
    deadline = time.monotonic() + timeout_seconds
    observations: list[str] = []
    while True:
        observed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        rows: list[tuple[int, int]] = []
        if probe.returncode == 0:
            for line in probe.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 2:
                    try:
                        rows.append((int(fields[0]), int(fields[1])))
                    except ValueError:
                        rows = []
                        break
        ready = len(rows) == 8 and [index for index, _ in rows] == list(range(8))
        ready = ready and all(used <= idle_memory_mib for _, used in rows)
        observations.append(f"{observed_at} ready={ready} rows={rows!r} rc={probe.returncode}")
        log.write_text("\n".join(observations) + "\n", encoding="utf-8")
        if ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"eight idle H20 GPUs unavailable after {timeout_seconds} seconds")
        time.sleep(30)


def main() -> int:
    """Evaluate one candidate on the exact DeepSeek-V4-Flash denominator.

    :return: Zero after writing a schema-valid CAKE evaluation document.
    :rtype: int
    """
    input_path = Path(os.environ["LOOM_EVALUATION_INPUT"])
    output_path = Path(os.environ["LOOM_EVALUATION_OUTPUT"])
    artifact_root = Path(os.environ["LOOM_EVALUATION_ARTIFACT_ROOT"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    request = json.loads(input_path.read_text(encoding="utf-8"))
    campaign = request["campaign"]
    lane = request["lane"]
    attempt = request["attempt"]
    candidate = request["candidate"]

    _wait_for_exclusive_gpus(artifact_root / "gpu-availability.log")

    shape_artifact = artifact_root / "dsv4-flash-shape.json"
    shape = {
        "model": "deepseek-v4-flash-mxfp4",
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_experts": 256,
        "num_topk": 6,
        "num_ranks": 8,
        "batches": list(FLASH_BATCHES),
        "pr383_us": PR383_US,
    }
    shape_artifact.write_text(json.dumps(shape, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    correctness_log = artifact_root / "correctness.log"
    correctness = _run(
        [
            sys.executable,
            "tests/test_mxfp4_mega_moe_sm90_correctness.py",
            "--batches", "8", "32",
            "--hidden", "4096",
            "--intermediate-hidden", "2048",
            "--num-experts", "256",
            "--num-topk", "6",
            "--num-processes", "8",
            "--num-max-tokens-per-rank", "8192",
            "--weight-scales", "0.05",
            "--global-scale-modes", "none", "expert",
        ],
        correctness_log,
    )

    benchmark_log = artifact_root / "paired-benchmark.log"
    benchmark = _run(
        [
            sys.executable,
            "tests/bench_mega_moe_formats_sm90.py",
            "--num-processes", "8",
            "--arms", "fp8", "mxfp4",
            "--baseline", "fp8",
            "--batches", *(str(m) for m in FLASH_BATCHES),
            "--hidden", "4096",
            "--intermediate-hidden", "2048",
            "--num-experts", "256",
            "--num-topk", "6",
            "--num-max-tokens-per-rank", "8192",
            "--fast-math", "1",
            "--num-tests", "20",
            "--reps", "3",
        ],
        benchmark_log,
    )

    rows: list[dict[str, float | int]] = []
    if benchmark.returncode == 0:
        for match in _ROW.finditer(benchmark.stdout):
            m = int(match.group("m"))
            fp8_us = float(match.group("fp8"))
            mxfp4_us = float(match.group("mxfp4"))
            rows.append({
                "m": m,
                "fp8_us": fp8_us,
                "mxfp4_us": mxfp4_us,
                "speedup_vs_live_fp8": fp8_us / mxfp4_us,
                "speedup_vs_pr383_published": PR383_US[m] / mxfp4_us,
            })
    rows.sort(key=lambda row: int(row["m"]))
    complete = [int(row["m"]) for row in rows] == list(FLASH_BATCHES)
    minimum = min((float(row["speedup_vs_live_fp8"]) for row in rows), default=0.0)
    geomean = (
        math.exp(sum(math.log(float(row["speedup_vs_live_fp8"])) for row in rows) / len(rows))
        if rows else 0.0
    )
    performance_pass = complete and minimum >= 1.05 and geomean >= 1.10
    perf_artifact = artifact_root / "paired-performance.json"
    perf_artifact.write_text(
        json.dumps({
            "rows": rows,
            "complete": complete,
            "minimum_speedup_vs_live_fp8": minimum,
            "geomean_speedup_vs_live_fp8": geomean,
            "required_minimum": 1.05,
            "required_geomean": 1.10,
            "benchmark_returncode": benchmark.returncode,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gates = [
        _gate("dsv4_flash_structure", True, shape_artifact, "Exact H=4096/IH=2048/E=256/topk=6/world=8 denominator."),
        _gate("mxfp4_correctness", correctness.returncode == 0, correctness_log, "Exact-dequantized MXFP4 correctness on M=8 and M=32."),
        _gate("paired_performance", performance_pass, perf_artifact, f"min={minimum:.6f}x geomean={geomean:.6f}x over {len(rows)}/11 rows."),
    ]
    comparable = correctness.returncode == 0 and complete
    document = {
        "schema_version": 20,
        "kind": "loom_kernel_candidate_evaluation",
        "campaign_id": campaign["campaign_id"],
        "wave": lane["wave"],
        "lane_id": attempt["attempt_id"],
        "candidate_id": candidate["candidate_id"],
        "commit": candidate["commit"],
        "evaluator_id": "sm90-mxfp4-megamoe-dsv4-flash-h20-v1",
        "gates": gates,
        "metric": {
            "name": "minimum_speedup_vs_baseline",
            "value": minimum,
            "unit": "ratio",
            "comparable": comparable,
            "artifact": "evidence/paired-performance.json",
            "artifact_sha256": _sha256(perf_artifact),
        },
        "summary": f"DeepSeek-V4-Flash H20: correctness_rc={correctness.returncode}, min={minimum:.6f}x, geomean={geomean:.6f}x.",
        "dispatcher_evidence": None,
        "dispatcher_target_evidence": None,
        "shape_performance_evidence": None,
        "seed_attribution_evidence": None,
        "seed_performance_evidence": None,
        "seed_performance_failure_evidence": None,
        "evaluation_path_provenance": None,
        "dispatcher_full_latency_evidence": None,
        "variance_audit_evidence": None,
        "correctness_failure_signature": None,
        "selected_isa_full_validation": None,
    }
    output_path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
