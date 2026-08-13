from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from medumm.core.config import find_project_root
from medumm.core.io import ensure_directory, write_json
from medumm.post_training.research_routes import ROUTES


ROUTE_ACCEPTANCE_ORDER = (
    "bagel_sft",
    "reca",
    "unicot",
    "irg",
    "unigame",
    "unipath",
    "latentumm",
)
_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")


def _snapshot_revision(project_root: Path) -> str:
    configured = os.environ.get("MEDUMM_SOURCE_COMMIT", "").strip()
    if _COMMIT.fullmatch(configured):
        return configured.lower()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "The source snapshot has no readable Git revision; set "
            "MEDUMM_SOURCE_COMMIT to the exact 40-character commit under test."
        ) from error
    revision = completed.stdout.strip()
    if not _COMMIT.fullmatch(revision):
        raise ValueError("Unable to determine a full MedUMM source revision.")
    return revision.lower()


def build_contract_smoke_config(
    method: str,
    *,
    project_root: Path,
    output_directory: Path,
    source_revision: str,
) -> dict[str, Any]:
    """Build one independently launchable public-CLI acceptance config."""

    if method not in ROUTES:
        raise KeyError(f"Unknown post-training route: {method!r}.")
    if not _COMMIT.fullmatch(source_revision):
        raise ValueError("source_revision must be a full 40-character commit.")
    stages: list[dict[str, Any]] = []
    for index, stage in enumerate(ROUTES[method].stages, 1):
        checkpoint = output_directory / "checkpoints" / f"{index:02d}-{stage.name}.json"
        launcher_args: dict[str, Any] = {
            "method": method,
            "stage": stage.name,
            "checkpoint_path": str(checkpoint),
            "steps": 6,
        }
        item: dict[str, Any] = {
            "name": stage.name,
            "launcher": {
                "entrypoint": "python",
                "cwd": str(project_root),
                "module": "medumm.post_training.contract_smoke_worker",
                "flag_style": "underscore",
                "bool_style": "value",
                "args": launcher_args,
            },
            "artifacts": {"checkpoint_path": str(checkpoint)},
        }
        if stage.depends_on:
            dependency = stage.depends_on[-1]
            reference = f"{{{{stages.{dependency}.checkpoint}}}}"
            item["inputs"] = {f"{dependency}_checkpoint": reference}
            launcher_args["previous_checkpoint"] = reference
        stages.append(item)
    return {
        "schema_version": "1.0",
        "runtime": {"seed": 42, "device": "cpu", "dtype": "float32"},
        "post_training": {
            "method": method,
            "execution": "launch",
            "acceptance_mode": "contract_smoke",
            "source": {
                "repository": "https://github.com/lixiangcog/MedUMM",
                "revision": source_revision,
                "license": "Apache-2.0",
                "code_root": str(project_root),
                "strict_git_revision": False,
            },
            "data": {
                "contract_manifest": str(
                    project_root
                    / "examples/medical/post_training/research_routes_contract.jsonl"
                ),
                "provenance": str(
                    project_root / "examples/medical/post_training/provenance.json"
                ),
                "license": "CC0-1.0",
                "deidentified": True,
            },
            "stages": stages,
            "output_directory": str(output_directory),
        },
    }


def _run_cli(command: list[str], *, project_root: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = (
        source_path + os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else source_path
    )
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(
        "$ " + shlex.join(command) + "\n\n"
        + completed.stdout
        + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"CLI exited with code {completed.returncode}; see {log_path}."
        )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _verify_one(
    method: str,
    *,
    project_root: Path,
    output_root: Path,
    source_revision: str,
    position: int,
    total: int,
) -> dict[str, Any]:
    route = ROUTES[method]
    route_output = ensure_directory(output_root / method)
    profile_output = route_output / "paper-profile-plan"
    profile_path = project_root / "configs/post_training/profiles" / f"{method}.yaml"

    print(
        f"[MedUMM][{position}/{total}] {method}: validating paper-runtime profile plan",
        flush=True,
    )
    profile_log = route_output / "paper-profile-plan-cli.log"
    _run_cli(
        [
            sys.executable,
            "-m",
            "medumm",
            "post-train",
            "--config",
            str(profile_path),
            "--plan",
            "--set",
            f"post_training.output_directory={profile_output}",
        ],
        project_root=project_root,
        log_path=profile_log,
    )
    profile_result = _read_object(profile_output / "result.json")
    profile_preflight = _read_object(profile_output / "preflight.json")
    expected_stages = [stage.name for stage in route.stages]
    planned_stages = profile_result.get("metadata", {}).get("stages")
    if profile_result.get("status") != "planned" or planned_stages != expected_stages:
        raise RuntimeError(f"{method} paper profile did not preserve its stage graph.")
    if profile_result.get("metadata", {}).get("ready") is not False:
        raise RuntimeError(f"{method} paper profile unexpectedly became launch-ready.")
    if profile_preflight.get("acceptance_mode") != "paper_runtime":
        raise RuntimeError(f"{method} paper profile used the wrong acceptance mode.")

    print(
        f"[MedUMM][{position}/{total}] {method}: launching {len(route.stages)} contract stage(s)",
        flush=True,
    )
    smoke_config = build_contract_smoke_config(
        method,
        project_root=project_root,
        output_directory=route_output,
        source_revision=source_revision,
    )
    smoke_config_path = write_json(route_output / "contract-smoke-config.json", smoke_config)
    smoke_log = route_output / "contract-smoke-cli.log"
    start = time.perf_counter()
    _run_cli(
        [
            sys.executable,
            "-m",
            "medumm",
            "post-train",
            "--config",
            str(smoke_config_path),
        ],
        project_root=project_root,
        log_path=smoke_log,
    )
    duration = time.perf_counter() - start
    smoke_result = _read_object(route_output / "result.json")
    smoke_preflight = _read_object(route_output / "preflight.json")
    if smoke_result.get("status") != "completed":
        raise RuntimeError(f"{method} contract execution did not complete.")
    if smoke_preflight.get("acceptance_mode") != "contract_smoke":
        raise RuntimeError(f"{method} contract execution used the wrong acceptance mode.")
    if smoke_result.get("metadata", {}).get("fidelity") != "contract_smoke":
        raise RuntimeError(f"{method} contract result made an invalid fidelity claim.")

    stage_evidence: list[dict[str, Any]] = []
    previous_checkpoint: str | None = None
    result_stages = smoke_result.get("metadata", {}).get("stages", [])
    if len(result_stages) != len(route.stages):
        raise RuntimeError(f"{method} completed an unexpected number of stages.")
    for expected, completed in zip(route.stages, result_stages):
        checkpoint_path = Path(str(completed.get("checkpoint_path", "")))
        log_path = route_output / expected.name / "training.log"
        if completed.get("stage") != expected.name or completed.get("status") != "completed":
            raise RuntimeError(f"{method}/{expected.name} did not complete in order.")
        if not checkpoint_path.is_file() or not log_path.is_file() or not log_path.stat().st_size:
            raise RuntimeError(f"{method}/{expected.name} is missing its checkpoint or log.")
        checkpoint = _read_object(checkpoint_path)
        if (
            checkpoint.get("method") != method
            or checkpoint.get("stage") != expected.name
            or checkpoint.get("paper_fidelity_claim") is not False
        ):
            raise RuntimeError(f"{method}/{expected.name} wrote an invalid checkpoint.")
        consumed_dependency = None
        if expected.depends_on:
            consumed_dependency = checkpoint.get("previous_checkpoint") == previous_checkpoint
            if not consumed_dependency:
                raise RuntimeError(
                    f"{method}/{expected.name} did not consume the preceding checkpoint."
                )
        stage_evidence.append(
            {
                "name": expected.name,
                "status": "completed",
                "checkpoint_path": str(checkpoint_path),
                "training_log": str(log_path),
                "consumed_dependency_checkpoint": consumed_dependency,
            }
        )
        previous_checkpoint = str(checkpoint_path)

    print(
        f"[MedUMM][PASS] {method}: {len(stage_evidence)} stage(s), {duration:.3f}s",
        flush=True,
    )
    return {
        "method": method,
        "display_name": route.display_name,
        "status": "passed",
        "paper_profile_plan": {
            "status": "passed",
            "launch_ready": False,
            "preflight_path": str(profile_output / "preflight.json"),
            "cli_log": str(profile_log),
        },
        "contract_execution": {
            "status": "passed",
            "duration_seconds": round(duration, 6),
            "preflight_path": str(route_output / "preflight.json"),
            "result_path": str(route_output / "result.json"),
            "cli_log": str(smoke_log),
            "stages": stage_evidence,
        },
        "paper_runtime": {
            "status": "not_run",
            "reason": "Requires pinned upstream checkout, model assets, audited medical data, and GPU resources.",
        },
    }


def verify_research_routes(
    *,
    project_root: Path,
    output_directory: Path,
    methods: Iterable[str] = ROUTE_ACCEPTANCE_ORDER,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Sequentially exercise every selected route through the public CLI."""

    root = project_root.resolve()
    output = ensure_directory(output_directory.resolve())
    requested = list(dict.fromkeys(methods))
    unknown = sorted(set(requested) - set(ROUTES))
    if unknown:
        raise KeyError(f"Unknown post-training route(s): {', '.join(unknown)}.")
    existing = [str(output / method) for method in requested if (output / method).exists()]
    if existing:
        raise FileExistsError(
            "Acceptance output must be fresh; choose a new output directory. "
            f"Existing route paths: {', '.join(existing)}"
        )
    revision = source_revision or _snapshot_revision(root)
    if not _COMMIT.fullmatch(revision):
        raise ValueError("source_revision must be a full 40-character commit.")

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for position, method in enumerate(requested, 1):
        try:
            results.append(
                _verify_one(
                    method,
                    project_root=root,
                    output_root=output,
                    source_revision=revision,
                    position=position,
                    total=len(requested),
                )
            )
        except Exception as error:  # keep checking remaining independent routes
            print(f"[MedUMM][FAIL] {method}: {error}", flush=True)
            results.append(
                {
                    "method": method,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "paper_runtime": {"status": "not_run"},
                }
            )

    passed = sum(result["status"] == "passed" for result in results)
    summary = {
        "schema_version": "1.0",
        "suite": "medumm_v1.1_sequential_post_training_acceptance",
        "status": "passed" if passed == len(results) else "failed",
        "source_revision": revision,
        "validation_scope": {
            "paper_profile_plan": "Stage graph and native launcher contract only.",
            "contract_execution": "Public CLI, subprocess, dependency, log, and checkpoint behavior.",
            "paper_runtime": "Not run by this dependency-light suite.",
        },
        "routes_requested": len(results),
        "routes_passed": passed,
        "stages_executed": sum(
            len(result.get("contract_execution", {}).get("stages", []))
            for result in results
        ),
        "duration_seconds": round(time.perf_counter() - started, 6),
        "routes": results,
        "clinical_use": False,
        "paper_fidelity_claim": False,
    }
    summary_path = write_json(output / "summary.json", summary)
    print(
        f"[MedUMM] sequential acceptance {summary['status']}: "
        f"{passed}/{len(results)} routes, {summary['stages_executed']} stages; {summary_path}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequentially validate MedUMM v1.1 post-training routes."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=find_project_root(Path.cwd()),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/post_training/v1.1-sequential-acceptance"),
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=ROUTE_ACCEPTANCE_ORDER,
        default=[],
        help="Validate one method; repeat for several. The default validates all seven.",
    )
    parser.add_argument("--source-revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = arguments.project_root.resolve()
    output = arguments.output_directory
    if not output.is_absolute():
        output = project_root / output
    summary = verify_research_routes(
        project_root=project_root,
        output_directory=output,
        methods=arguments.method or ROUTE_ACCEPTANCE_ORDER,
        source_revision=arguments.source_revision,
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
