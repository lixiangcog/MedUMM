from importlib.util import module_from_spec, spec_from_file_location

from tests.conftest import PROJECT_ROOT


def _module():
    path = PROJECT_ROOT / "scripts/prepare_vqa_rad.py"
    spec = spec_from_file_location("prepare_vqa_rad", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vqa_rad_export_uses_stable_identifiers():
    module = _module()
    assert module._record_id("test", 3, "Is this normal?") == module._record_id(
        "test", 3, "Is this normal?"
    )
    assert module._record_id("test", 3, "Is this normal?").startswith("vqa-rad-test-00003-")


def test_vqa_rad_script_parser_accepts_closed_only(tmp_path):
    module = _module()
    values = module.build_parser().parse_args(
        ["--output-directory", str(tmp_path), "--closed-only", "--max-samples", "2"]
    )
    assert values.closed_only
    assert values.max_samples == 2
