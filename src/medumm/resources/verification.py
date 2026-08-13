from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from medumm.resources.specs import DATASET_RESOURCES, MODEL_RESOURCES


REACHABLE_HTTP_CODES = {200, 202, 206, 301, 302, 303, 307, 308, 401, 403, 429}


def _probe(url: str, timeout: float) -> dict[str, Any]:
    code: int | None = None
    resolved: str | None = None
    last_error: Exception | None = None
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(
            url,
            method=method,
            headers={
                "User-Agent": "MedUMM-resource-audit/0.8",
                "Range": "bytes=0-0",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
                context=ssl.create_default_context(),
            ) as response:
                code = int(response.status)
                resolved = str(response.url)
                break
        except urllib.error.HTTPError as error:
            code = int(error.code)
            resolved = str(error.url)
            if code in REACHABLE_HTTP_CODES or code != 405:
                break
            last_error = error
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if method == "GET":
                break
    if code is None:
        return {
            "url": url,
            "reachable": False,
            "status_code": None,
            "resolved_url": None,
            "error": f"{type(last_error).__name__}: {last_error}",
        }
    return {
        "url": url,
        "reachable": code in REACHABLE_HTTP_CODES,
        "status_code": code,
        "resolved_url": resolved,
        "error": None if code in REACHABLE_HTTP_CODES else f"HTTP {code}",
    }


def verify_sources(
    *,
    kind: str = "all",
    fields: Iterable[str] = ("source",),
    timeout: float = 10.0,
    workers: int = 8,
) -> dict[str, Any]:
    """Probe catalog URLs without downloading model weights or datasets."""

    normalized_kind = kind.strip().lower()
    if normalized_kind not in {"all", "model", "dataset"}:
        raise ValueError("Resource kind must be model, dataset, or all.")
    selected_fields = tuple(dict.fromkeys(str(item) for item in fields))
    if not selected_fields or set(selected_fields) - {"source", "paper", "official_code"}:
        raise ValueError("Verification fields must be source, paper, or official_code.")
    selected: list[tuple[str, Any]] = []
    if normalized_kind in {"all", "model"}:
        selected.extend(("model", item) for item in MODEL_RESOURCES.values())
    if normalized_kind in {"all", "dataset"}:
        selected.extend(("dataset", item) for item in DATASET_RESOURCES.values())
    targets = [
        (resource_kind, item.name, field, str(getattr(item, field)))
        for resource_kind, item in selected
        for field in selected_fields
        if getattr(item, field) is not None
    ]
    urls = list(dict.fromkeys(url for _, _, _, url in targets))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        probes = dict(zip(urls, executor.map(lambda url: _probe(url, timeout), urls)))
    rows = [
        {
            "kind": resource_kind,
            "name": name,
            "field": field,
            **probes[url],
        }
        for resource_kind, name, field, url in targets
    ]
    failures = [row for row in rows if not row["reachable"]]
    return {
        "schema_version": "1.0",
        "catalog_version": MODEL_RESOURCES.version,
        "scope": "url_reachability_only",
        "checked": len(rows),
        "passed": len(rows) - len(failures),
        "failed": len(failures),
        "valid": not failures,
        "results": rows,
    }
