from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from medumm.core.interfaces import PostTrainer
from medumm.core.io import ensure_directory, read_jsonl, write_json
from medumm.core.results import Artifact, TrainingResult
from medumm.core.runtime import RuntimeContext, environment_snapshot, redact_secrets


_UNEXPANDED_ENV = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_STAGE_REFERENCE = re.compile(r"\{\{stages\.([a-z0-9_]+)\.checkpoint\}\}")
_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_SENSITIVE_ARGUMENTS = frozenset(
    {"api_key", "access_token", "auth_token", "hf_token", "password", "secret", "token"}
)


@dataclass(frozen=True, slots=True)
class RouteStage:
    name: str
    objective: str
    required_sample_fields: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    default_entrypoint: str = "torchrun"
    default_flag_style: str = "underscore"
    default_bool_style: str = "value"


@dataclass(frozen=True, slots=True)
class ResearchRoute:
    name: str
    display_name: str
    summary: str
    paper_url: str
    code_url: str
    fidelity: str
    stages: tuple[RouteStage, ...]
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["stage_order"] = [stage.name for stage in self.stages]
        return value


ROUTES: dict[str, ResearchRoute] = {
    "bagel_sft": ResearchRoute(
        name="bagel_sft",
        display_name="BAGEL SFT",
        summary="Joint supervised fine-tuning over understanding and visual generation data.",
        paper_url="https://arxiv.org/abs/2505.14683",
        code_url="https://github.com/ByteDance-Seed/Bagel",
        fidelity="official_runtime_bridge",
        aliases=("sft",),
        stages=(
            RouteStage(
                "joint_sft",
                "masked text cross-entropy plus visual generation loss",
                ("id", "task", "prompt", "response"),
            ),
        ),
    ),
    "reca": ResearchRoute(
        name="reca",
        display_name="RecA",
        summary="Reconstruct an image from its frozen understanding-encoder representation.",
        paper_url="https://arxiv.org/abs/2509.07295",
        code_url="https://github.com/HorizonWind2004/reconstruction-alignment",
        fidelity="official_runtime_bridge",
        stages=(
            RouteStage(
                "reconstruction_alignment",
                "self-supervised image reconstruction conditioned on understanding embeddings",
                ("id", "image", "reconstruction_target"),
            ),
        ),
    ),
    "unicot": ResearchRoute(
        name="unicot",
        display_name="Uni-CoT",
        summary="Hierarchical text-and-vision chain-of-thought supervised fine-tuning.",
        paper_url="https://arxiv.org/abs/2508.05606",
        code_url="https://github.com/Fr0zenCrane/UniCoT",
        fidelity="official_runtime_bridge",
        stages=(
            RouteStage(
                "hierarchical_cot_sft",
                "macro trajectory supervision plus micro transition objectives",
                ("id", "prompt", "trajectory", "response"),
            ),
        ),
    ),
    "irg": ResearchRoute(
        name="irg",
        display_name="IRG",
        summary="Two-stage interleaved reasoning, generation, reflection, and refinement.",
        paper_url="https://arxiv.org/abs/2509.06945",
        code_url="https://github.com/Osilly/Interleaving-Reasoning-Generation",
        fidelity="official_runtime_bridge",
        stages=(
            RouteStage(
                "think_generate",
                "initial textual reasoning and image generation",
                ("id", "prompt", "reasoning", "initial_image"),
            ),
            RouteStage(
                "reflect_refine",
                "textual reflection and faithful image refinement",
                ("id", "prompt", "initial_image", "reflection", "refined_image"),
                depends_on=("think_generate",),
            ),
        ),
    ),
    "unigame": ResearchRoute(
        name="unigame",
        display_name="UniGame",
        summary="Minimax self-play between a latent perturber and the understanding branch.",
        paper_url="https://arxiv.org/abs/2511.19413",
        code_url=(
            "https://github.com/AIFrontierLab/TorchUMM/tree/main/"
            "src/umm/post_training/unigame"
        ),
        fidelity="reference_runtime_bridge",
        metadata={
            "archived_standalone_code": "https://github.com/AIFrontierLab/UniGame"
        },
        stages=(
            RouteStage(
                "self_adversarial",
                "decoder-constrained challenge step and hard-example understanding step",
                ("id", "image", "question", "answer"),
            ),
        ),
    ),
    "unipath": ResearchRoute(
        name="unipath",
        display_name="UniPath",
        summary="Path-conditioned executor training followed by input-adaptive route planning.",
        paper_url="https://arxiv.org/abs/2605.11400",
        code_url=(
            "https://github.com/AIFrontierLab/TorchUMM/tree/main/"
            "src/umm/post_training/unipath"
        ),
        fidelity="reference_runtime_bridge",
        metadata={"paths": ["direct", "l0", "l1", "l2", "l3"]},
        stages=(
            RouteStage(
                "executor_understanding_text",
                "text-only path-conditioned trajectory imitation",
                ("id", "prompt", "reasoning_path", "trajectory", "answer"),
            ),
            RouteStage(
                "executor_understanding_visual",
                "visual-thought trajectory imitation",
                ("id", "prompt", "reasoning_path", "trajectory", "answer", "image"),
                depends_on=("executor_understanding_text",),
            ),
            RouteStage(
                "executor_image_answer_plain",
                "plain image-answer trajectory with text and latent reconstruction losses",
                ("id", "prompt", "reasoning_path", "trajectory", "answer", "image"),
                depends_on=("executor_understanding_visual",),
            ),
            RouteStage(
                "executor_image_answer_visual",
                "visual image-answer trajectory with visual-summary consistency",
                ("id", "prompt", "reasoning_path", "trajectory", "answer", "image"),
                depends_on=("executor_image_answer_plain",),
            ),
            RouteStage(
                "planner",
                "multi-label path-outcome prediction with cost-aware route selection",
                ("id", "planner_features", "path_outcomes"),
                depends_on=("executor_image_answer_visual",),
                default_entrypoint="python",
                default_flag_style="hyphen",
                default_bool_style="flag",
            ),
        ),
    ),
    "latentumm": ResearchRoute(
        name="latentumm",
        display_name="LatentUMM",
        summary="Dual latent alignment followed by stochastic latent-dynamics stabilization.",
        paper_url="https://arxiv.org/abs/2605.17766",
        code_url=(
            "https://github.com/AIFrontierLab/TorchUMM/tree/main/"
            "src/umm/post_training/LatentUMM"
        ),
        fidelity="reference_runtime_bridge",
        stages=(
            RouteStage(
                "dual_latent_alignment",
                "cross-modal and bidirectional capacity alignment",
                ("id", "text_embedding", "image_embedding"),
                default_entrypoint="python",
                default_flag_style="hyphen",
                default_bool_style="flag",
            ),
            RouteStage(
                "latent_dynamics",
                "stochastic latent rollouts and trajectory preference optimization",
                ("id", "text_embedding", "image_embedding", "preferred_latent", "rejected_latent"),
                depends_on=("dual_latent_alignment",),
            ),
        ),
    ),
}


ROUTE_ALIASES = {
    alias: route.name for route in ROUTES.values() for alias in route.aliases
}


def research_route_catalog() -> list[dict[str, Any]]:
    """Return a serializable, dependency-free description of every route."""

    return [ROUTES[name].to_dict() for name in sorted(ROUTES)]


def _merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _merge(base[key], value) if key in base else value
        return merged
    return override


def route_template(method: str) -> dict[str, Any]:
    """Build a safe plan-only template for one post-training route."""

    normalized = ROUTE_ALIASES.get(method.casefold(), method.casefold())
    route = ROUTES[normalized]
    stages: list[dict[str, Any]] = []
    for stage in route.stages:
        stages.append(
            {
                "name": stage.name,
                "data": {
                    "contract_manifest": "REPLACE_WITH_DEIDENTIFIED_AUDIT_JSONL",
                    "deidentified": False,
                    "license": "REPLACE_WITH_DATASET_LICENSE",
                    "provenance": "REPLACE_WITH_PROVENANCE_JSON",
                },
                "launcher": {
                    "entrypoint": stage.default_entrypoint,
                    "cwd": "REPLACE_WITH_PINNED_SOURCE_CHECKOUT",
                    "script": "REPLACE_WITH_OFFICIAL_TRAINING_SCRIPT",
                    "args": {},
                    "flag_style": stage.default_flag_style,
                    "bool_style": stage.default_bool_style,
                },
                "artifacts": {
                    "checkpoint_path": f"outputs/post_training/{route.name}/{stage.name}/checkpoint"
                },
            }
        )
        if stage.depends_on:
            dependency = stage.depends_on[-1]
            reference = f"{{{{stages.{dependency}.checkpoint}}}}"
            stages[-1]["inputs"] = {f"{dependency}_checkpoint": reference}
            stages[-1]["launcher"]["args"]["previous_checkpoint"] = reference
    return {
        "schema_version": "1.0",
        "runtime": {"seed": 42, "device": "cuda", "dtype": "bfloat16"},
        "post_training": {
            "method": route.name,
            "execution": "plan",
            "source": {
                "repository": route.code_url,
                "revision": "REPLACE_WITH_40_CHARACTER_COMMIT",
                "license": "VERIFY_AT_PINNED_REVISION",
                "code_root": "REPLACE_WITH_PINNED_SOURCE_CHECKOUT",
                "strict_git_revision": True,
            },
            "stages": stages,
            "output_directory": f"outputs/post_training/{route.name}",
        },
    }


def _contains_unresolved(value: Any) -> bool:
    if isinstance(value, str):
        return bool(
            _UNEXPANDED_ENV.search(value)
            or value.startswith(("REPLACE_WITH_", "VERIFY_"))
        )
    if isinstance(value, list):
        return any(_contains_unresolved(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_unresolved(item) for item in value.values())
    return False


def _project_path(value: Any, root: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else root / path


def _git_revision(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _git_dirty(path: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def _resolve_stage_references(value: Any, checkpoints: dict[str, str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in checkpoints:
                raise ValueError(f"Stage checkpoint {name!r} is not available yet.")
            return checkpoints[name]

        return _STAGE_REFERENCE.sub(replace, value)
    if isinstance(value, list):
        return [_resolve_stage_references(item, checkpoints) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_stage_references(item, checkpoints) for key, item in value.items()}
    return value


def _format_flag(key: str, style: str) -> str:
    if style == "hyphen":
        return "--" + key.replace("_", "-")
    if style == "underscore":
        return "--" + key.replace("-", "_")
    if style == "raw":
        return "--" + key
    raise ValueError("launcher.flag_style must be hyphen, underscore, or raw.")


def _argument_tokens(arguments: dict[str, Any], launcher: dict[str, Any]) -> list[str]:
    bool_style = str(launcher.get("bool_style", "value"))
    if bool_style not in {"value", "flag"}:
        raise ValueError("launcher.bool_style must be value or flag.")
    compact = {str(item) for item in launcher.get("compact_list_args", [])}
    tokens: list[str] = []
    for key, value in arguments.items():
        if value is None:
            continue
        flag = _format_flag(str(key), str(launcher.get("flag_style", "underscore")))
        if isinstance(value, bool):
            if bool_style == "value":
                tokens.extend([flag, "True" if value else "False"])
            elif value:
                tokens.append(flag)
        elif isinstance(value, (list, tuple)) and str(key) in compact:
            tokens.append(flag)
            tokens.extend(str(item) for item in value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                tokens.extend([flag, str(item)])
        else:
            tokens.extend([flag, str(value)])
    return tokens


def _launcher_command(launcher: dict[str, Any], cwd: Path) -> list[str]:
    entrypoint = str(launcher.get("entrypoint", "torchrun")).casefold()
    module = str(launcher.get("module", "")).strip()
    script = str(launcher.get("script", "")).strip()
    callable_target = str(launcher.get("callable", "")).strip()
    if sum(map(bool, (module, script, callable_target))) != 1:
        raise ValueError("launcher requires exactly one of module, script, or callable.")
    if entrypoint not in {"python", "torchrun", "bash", "callable"}:
        raise ValueError(
            "launcher.entrypoint must be python, torchrun, bash, or callable."
        )
    if entrypoint == "bash" and module:
        raise ValueError("A bash launcher cannot execute a Python module.")
    if entrypoint == "callable":
        if not callable_target or module or script:
            raise ValueError(
                "A callable launcher requires launcher.callable and no module/script."
            )
        arguments = launcher.get("args", {}) or {}
        if not isinstance(arguments, dict):
            raise ValueError("launcher.args must be a mapping.")
        return [
            sys.executable,
            "-m",
            "medumm.post_training.callable_worker",
            "--target",
            callable_target,
            "--config-json",
            json.dumps(arguments),
        ]
    if callable_target:
        raise ValueError("launcher.callable requires entrypoint=callable.")

    if entrypoint == "python":
        command = [sys.executable]
    elif entrypoint == "bash":
        command = ["bash"]
    else:
        torchrun = shutil.which("torchrun")
        command = [torchrun] if torchrun else [sys.executable, "-m", "torch.distributed.run"]
        distributed = launcher.get("torchrun", {}) or {}
        if not isinstance(distributed, dict):
            raise ValueError("launcher.torchrun must be a mapping.")
        defaults = {"nnodes": 1, "node_rank": 0, "nproc_per_node": 1}
        for key, default in defaults.items():
            command.append(f"--{key}={distributed.get(key, default)}")
        for key in ("master_addr", "master_port", "rdzv_backend", "rdzv_endpoint"):
            if distributed.get(key) is not None:
                command.append(f"--{key}={distributed[key]}")
        command.extend(str(item) for item in distributed.get("extra_args", []))

    if module:
        command.extend(["-m", module])
    else:
        script_path = Path(script)
        script_path = script_path if script_path.is_absolute() else cwd / script_path
        command.append(str(script_path.resolve()))
    arguments = launcher.get("args", {}) or {}
    if not isinstance(arguments, dict):
        raise ValueError("launcher.args must be a mapping.")
    command.extend(_argument_tokens(arguments, launcher))
    command.extend(str(item) for item in launcher.get("extra_args", []))
    return command


def _redacted_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for index, token in enumerate(redacted[:-1]):
        key = token.lstrip("-").replace("-", "_").casefold()
        if key in _SENSITIVE_ARGUMENTS or key.endswith(("_token", "_secret", "_password")):
            redacted[index + 1] = "<redacted>"
        if token == "--config-json":
            redacted[index + 1] = "<redacted-config>"
    return redacted


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_data(
    data: dict[str, Any], stage: RouteStage, root: Path, *, require_files: bool
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("deidentified") is not True:
        errors.append("data.deidentified must be true after a documented privacy review")
    if not str(data.get("license", "")).strip():
        errors.append("data.license is required")
    provenance_raw = data.get("provenance")
    manifest_raw = data.get("contract_manifest")
    if not provenance_raw:
        errors.append("data.provenance is required")
    if not manifest_raw:
        errors.append("data.contract_manifest is required")

    rows: list[dict[str, Any]] = []
    fingerprint = None
    manifest_path = None
    if manifest_raw and not _contains_unresolved(manifest_raw):
        manifest_path = _project_path(manifest_raw, root)
        if manifest_path.is_file():
            try:
                rows = read_jsonl(manifest_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"invalid contract manifest: {error}")
            else:
                fingerprint = _fingerprint(manifest_path)
                for index, row in enumerate(rows, 1):
                    missing = [field for field in stage.required_sample_fields if row.get(field) in (None, "")]
                    if missing:
                        errors.append(f"contract row {index} missing: {', '.join(missing)}")
        elif require_files:
            errors.append(f"contract manifest not found: {manifest_path}")
        else:
            warnings.append(f"contract manifest is not locally available: {manifest_path}")
    elif manifest_raw:
        warnings.append("contract manifest contains a template or unresolved environment value")

    if provenance_raw and not _contains_unresolved(provenance_raw):
        provenance_path = _project_path(provenance_raw, root)
        if require_files and not provenance_path.is_file():
            errors.append(f"provenance file not found: {provenance_path}")
        elif not provenance_path.is_file():
            warnings.append(f"provenance file is not locally available: {provenance_path}")
    elif provenance_raw:
        warnings.append("provenance contains a template or unresolved environment value")

    return {
        "status": "failed" if errors else "passed",
        "stage": stage.name,
        "required_sample_fields": list(stage.required_sample_fields),
        "manifest": str(manifest_path) if manifest_path else str(manifest_raw or ""),
        "fingerprint_sha256": fingerprint,
        "rows_audited": len(rows),
        "errors": errors,
        "warnings": warnings,
    }


def _normalize_stages(config: dict[str, Any], route: ResearchRoute) -> list[dict[str, Any]]:
    raw_stages = config.get("stages")
    if raw_stages is None:
        name = str(config.get("stage") or (route.stages[0].name if len(route.stages) == 1 else ""))
        if not name:
            raise ValueError(f"{route.name} requires stage or stages.")
        return [{**config, "name": name}]
    if not isinstance(raw_stages, list) or not raw_stages or not all(
        isinstance(item, dict) for item in raw_stages
    ):
        raise ValueError("post_training.stages must be a non-empty list of mappings.")
    inherited = {
        key: config[key]
        for key in ("source", "data", "launcher")
        if key in config
    }
    stages = [_merge(inherited, item) for item in raw_stages]
    names = [str(stage.get("name", "")) for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError("post_training.stages contains duplicate stage names.")
    return stages


def _validated_source(
    source: dict[str, Any], route: ResearchRoute, acceptance_mode: str
) -> str:
    repository = str(source.get("repository", "")).strip().rstrip("/")
    if acceptance_mode == "contract_smoke":
        expected = "https://github.com/lixiangcog/MedUMM"
        if repository.casefold() != expected.casefold():
            raise ValueError(
                f"contract_smoke source.repository must be {expected!r}; got {repository!r}."
            )
        return repository
    expected = route.code_url.split("/tree/", 1)[0].rstrip("/")
    if repository.casefold() != expected.casefold():
        raise ValueError(
            f"{route.name} source.repository must be {expected!r}; got {repository!r}."
        )
    return repository


class ResearchRouteTrainer(PostTrainer):
    """Validated external-runtime bridge for heterogeneous post-training methods."""

    def __init__(self, route_name: str) -> None:
        canonical = ROUTE_ALIASES.get(route_name.casefold(), route_name.casefold())
        self.route = ROUTES[canonical]
        self.name = canonical

    def fit(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        runtime: RuntimeContext,
    ) -> TrainingResult:
        execution = str(config.get("execution", "launch")).casefold()
        if execution not in {"plan", "launch"}:
            raise ValueError("post_training.execution must be plan or launch.")
        acceptance_mode = str(config.get("acceptance_mode", "paper_runtime")).casefold()
        if acceptance_mode not in {"paper_runtime", "contract_smoke"}:
            raise ValueError(
                "post_training.acceptance_mode must be paper_runtime or contract_smoke."
            )
        root = runtime.project_root
        output = _project_path(
            config.get("output_directory", f"outputs/post_training/{self.name}"), root
        )
        ensure_directory(output)
        source = config.get("source", {}) or {}
        if not isinstance(source, dict):
            raise ValueError("post_training.source must be a mapping.")
        source_revision = str(source.get("revision", "")).strip()
        source_root_raw = source.get("code_root")
        source_root = (
            _project_path(source_root_raw, root)
            if source_root_raw and not _contains_unresolved(source_root_raw)
            else None
        )
        source_errors: list[str] = []
        source_warnings: list[str] = []
        if not str(source.get("repository", "")).strip():
            source_errors.append("source.repository is required")
        else:
            try:
                _validated_source(source, self.route, acceptance_mode)
            except ValueError as error:
                source_errors.append(str(error))
        if not _COMMIT.fullmatch(source_revision):
            source_errors.append("source.revision must be a full 40-character commit")
        source_license = str(source.get("license", "")).strip()
        if not source_license or _contains_unresolved(source_license):
            source_errors.append("source.license must be a verified concrete license")
        if (
            acceptance_mode == "paper_runtime"
            and execution == "launch"
            and source.get("strict_git_revision") is not True
        ):
            source_errors.append(
                "paper_runtime launch requires source.strict_git_revision=true"
            )
        if source_root is None:
            source_errors.append("source.code_root must resolve to a pinned checkout")
        elif not source_root.is_dir():
            message = f"source checkout not found: {source_root}"
            (source_errors if execution == "launch" else source_warnings).append(message)
        elif source.get("strict_git_revision", True):
            actual = _git_revision(source_root)
            if actual != source_revision:
                message = f"source revision mismatch: configured={source_revision}, actual={actual}"
                (source_errors if execution == "launch" else source_warnings).append(message)
            dirty = _git_dirty(source_root)
            if dirty is True:
                message = f"source checkout has tracked modifications: {source_root}"
                (source_errors if execution == "launch" else source_warnings).append(message)
            elif dirty is None:
                message = f"source checkout Git status is unavailable: {source_root}"
                (source_errors if execution == "launch" else source_warnings).append(message)

        stage_by_name = {stage.name: stage for stage in self.route.stages}
        requested = _normalize_stages(config, self.route)
        plans: list[dict[str, Any]] = []
        checkpoints: dict[str, str] = {}
        completed_names: set[str] = set()
        for raw_stage in requested:
            stage_name = str(raw_stage.get("name", ""))
            if stage_name not in stage_by_name:
                available = ", ".join(stage_by_name)
                raise ValueError(f"Unknown {self.name} stage {stage_name!r}; available: {available}.")
            spec = stage_by_name[stage_name]
            missing_dependencies = [name for name in spec.depends_on if name not in completed_names]
            external_inputs = raw_stage.get("inputs", {}) or {}
            if not isinstance(external_inputs, dict):
                raise ValueError(f"Stage {stage_name} inputs must be a mapping.")
            for dependency in list(missing_dependencies):
                if external_inputs.get(f"{dependency}_checkpoint"):
                    missing_dependencies.remove(dependency)
            if missing_dependencies:
                raise ValueError(
                    f"Stage {stage_name} requires earlier stage(s) or explicit inputs: "
                    + ", ".join(missing_dependencies)
                )

            try:
                resolved = _resolve_stage_references(raw_stage, checkpoints)
            except ValueError:
                if execution != "plan":
                    raise
                resolved = raw_stage
            data = resolved.get("data", {}) or {}
            launcher = resolved.get("launcher", {}) or {}
            artifacts = resolved.get("artifacts", {}) or {}
            if not isinstance(data, dict) or not isinstance(launcher, dict) or not isinstance(artifacts, dict):
                raise ValueError(f"Stage {stage_name} data, launcher, and artifacts must be mappings.")
            audit = _audit_data(data, spec, root, require_files=execution == "launch")
            stage_output = _project_path(
                resolved.get("output_directory", output / stage_name), root
            )
            checkpoint_raw = artifacts.get("checkpoint_path")
            checkpoint = (
                _project_path(checkpoint_raw, root) if checkpoint_raw and not _contains_unresolved(checkpoint_raw) else None
            )
            for dependency in spec.depends_on:
                input_key = f"{dependency}_checkpoint"
                input_value = (resolved.get("inputs", {}) or {}).get(input_key)
                if (
                    dependency not in completed_names
                    and input_value
                    and not _contains_unresolved(input_value)
                ):
                    dependency_path = _project_path(input_value, root)
                    if execution == "launch" and not dependency_path.exists():
                        audit["errors"].append(
                            f"dependency checkpoint not found: {dependency_path}"
                        )
                        audit["status"] = "failed"
            cwd_raw = launcher.get("cwd", source_root or root)
            cwd = _project_path(cwd_raw, root) if not _contains_unresolved(cwd_raw) else root
            unresolved = _contains_unresolved(resolved)
            command_error = None
            try:
                command = _launcher_command(launcher, cwd)
            except (TypeError, ValueError) as error:
                command = []
                command_error = str(error)
            stage_errors = list(audit["errors"])
            if command_error:
                stage_errors.append(command_error)
            if checkpoint is None:
                stage_errors.append("artifacts.checkpoint_path must resolve to a concrete path")
            if acceptance_mode == "contract_smoke" and launcher.get("module") != (
                "medumm.post_training.contract_smoke_worker"
            ):
                stage_errors.append(
                    "contract_smoke may only launch medumm.post_training.contract_smoke_worker"
                )
            if execution == "launch":
                if unresolved:
                    stage_errors.append("stage contains unresolved environment or template values")
                if not cwd.is_dir():
                    stage_errors.append(f"launcher.cwd not found: {cwd}")
                if launcher.get("script"):
                    script_path = Path(str(launcher["script"]))
                    script_path = script_path if script_path.is_absolute() else cwd / script_path
                    if not script_path.is_file():
                        stage_errors.append(f"launcher script not found: {script_path}")

            plan = {
                "stage": stage_name,
                "objective": spec.objective,
                "depends_on": list(spec.depends_on),
                "status": "blocked" if stage_errors else "ready",
                "errors": stage_errors,
                "data_audit": audit,
                "cwd": str(cwd),
                "command": _redacted_command(command),
                "command_text": shlex.join(_redacted_command(command)) if command else "",
                "output_directory": str(stage_output),
                "checkpoint_path": str(checkpoint) if checkpoint else None,
            }
            plans.append(plan)
            if checkpoint:
                checkpoints[stage_name] = str(checkpoint)
            completed_names.add(stage_name)

        preflight = {
            "schema_version": "1.0",
            "method": self.name,
            "display_name": self.route.display_name,
            "execution": execution,
            "acceptance_mode": acceptance_mode,
            "fidelity": self.route.fidelity,
            "paper_url": self.route.paper_url,
            "code_url": self.route.code_url,
            "source": redact_secrets(source),
            "source_status": "failed" if source_errors else "passed",
            "source_errors": source_errors,
            "source_warnings": source_warnings,
            "stages": plans,
        }
        preflight_path = write_json(output / "preflight.json", preflight)
        artifacts_out = [Artifact("post_training_preflight", str(preflight_path), "application/json")]
        if execution == "plan":
            result = TrainingResult(
                method=self.name,
                status="planned",
                output_directory=str(output),
                artifacts=artifacts_out,
                metadata={
                    "fidelity": (
                        "contract_smoke" if acceptance_mode == "contract_smoke" else self.route.fidelity
                    ),
                    "stages": [plan["stage"] for plan in plans],
                    "ready": not source_errors and all(plan["status"] == "ready" for plan in plans),
                    "clinical_use": False,
                    "run_id": runtime.run_id,
                },
            )
            result_path = write_json(output / "result.json", result.to_dict())
            result.artifacts.append(Artifact("training_result", str(result_path), "application/json"))
            return result

        errors = source_errors + [
            f"{plan['stage']}: {error}" for plan in plans for error in plan["errors"]
        ]
        if errors:
            raise ValueError("Post-training preflight failed: " + "; ".join(errors))

        stage_results: list[dict[str, Any]] = []
        start = time.perf_counter()
        for plan, raw_stage in zip(plans, requested):
            resolved = _resolve_stage_references(raw_stage, checkpoints)
            launcher = resolved["launcher"]
            cwd = Path(plan["cwd"])
            command = _launcher_command(launcher, cwd)
            ensure_directory(plan["output_directory"])
            log_path = Path(plan["output_directory"]) / "training.log"
            env = os.environ.copy()
            environment = launcher.get("env", {}) or {}
            if not isinstance(environment, dict):
                raise ValueError(f"Stage {plan['stage']} launcher.env must be a mapping.")
            for key, value in environment.items():
                if value is None:
                    continue
                env_key = str(key)
                env_value = str(value)
                if env_key == "PYTHONPATH" and env.get(env_key):
                    env_value = env_value + os.pathsep + env[env_key]
                env[env_key] = env_value
            stage_start = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            duration = time.perf_counter() - stage_start
            if completed.returncode:
                raise RuntimeError(
                    f"{self.name}/{plan['stage']} exited with code {completed.returncode}; see {log_path}"
                )
            checkpoint = Path(str(plan["checkpoint_path"]))
            if not checkpoint.exists():
                raise RuntimeError(
                    f"{self.name}/{plan['stage']} finished without declared checkpoint: {checkpoint}"
                )
            checkpoints[plan["stage"]] = str(checkpoint)
            artifacts_out.extend(
                [
                    Artifact("training_log", str(log_path), "text/plain", {"stage": plan["stage"]}),
                    Artifact("checkpoint", str(checkpoint), metadata={"stage": plan["stage"]}),
                ]
            )
            stage_results.append(
                {
                    "stage": plan["stage"],
                    "status": "completed",
                    "duration_seconds": round(duration, 6),
                    "checkpoint_path": str(checkpoint),
                }
            )

        final_checkpoint = stage_results[-1]["checkpoint_path"]
        duration = time.perf_counter() - start
        result = TrainingResult(
            method=self.name,
            status="completed",
            output_directory=str(output),
            checkpoint_path=final_checkpoint,
            metrics={"stages_completed": float(len(stage_results)), "duration_seconds": duration},
            artifacts=artifacts_out,
            metadata={
                "fidelity": (
                    "contract_smoke" if acceptance_mode == "contract_smoke" else self.route.fidelity
                ),
                "stages": stage_results,
                "source_revision": source_revision,
                "environment": environment_snapshot(runtime),
                "clinical_use": False,
                "run_id": runtime.run_id,
            },
        )
        result_path = write_json(output / "result.json", result.to_dict())
        result.artifacts.append(Artifact("training_result", str(result_path), "application/json"))
        return result
