from importlib.util import module_from_spec, spec_from_file_location

from tests.conftest import PROJECT_ROOT


def _module():
    path = PROJECT_ROOT / "scripts/prepare_vqa_rad_tasks.py"
    spec = spec_from_file_location("prepare_vqa_rad_tasks", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vqa_rad_medical_task_mapping_rules_are_ordered_and_explainable():
    module = _module()
    cases = [
        ("Is there a pleural effusion?", "yes", "finding_assessment"),
        ("Describe the pathology in this image", "opacity", "clinical_description"),
        ("Where is the lesion located?", "left lung", "anatomy_localization"),
        ("How large is the mass?", "2 cm", "quantitative_assessment"),
        ("How was this image taken?", "CT", "image_context"),
        ("What is the most likely diagnosis?", "pneumonia", "diagnostic_reasoning"),
        (
            "This image is consistent with what condition?",
            "appendicitis",
            "diagnostic_reasoning",
        ),
    ]
    for question, answer, expected in cases:
        task, rule = module.classify_question(question, answer)
        assert task == expected
        assert rule.endswith("pattern") or rule.endswith("fallback")


def test_vqa_rad_task_export_parser_defaults_to_balanced_real_slice(tmp_path):
    module = _module()
    values = module.build_parser().parse_args(["--output-directory", str(tmp_path)])
    assert values.samples_per_task == 4
    assert values.revision == module.DEFAULT_REVISION
    assert len(module.REAL_TASKS) == 6


def test_case_identifier_depends_on_image_content():
    module = _module()

    class Image:
        size = (2, 2)

        def __init__(self, value):
            self.value = value

        def convert(self, mode):
            assert mode == "RGB"
            return self

        def tobytes(self):
            return bytes([self.value]) * 12

    assert module._case_id(Image(1)) == module._case_id(Image(1))
    assert module._case_id(Image(1)) != module._case_id(Image(2))
