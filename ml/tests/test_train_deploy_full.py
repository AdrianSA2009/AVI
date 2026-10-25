import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "train_deploy_full.py"
SPEC = importlib.util.spec_from_file_location("train_deploy_full", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_static26_class_map_preserves_jz_ids():
    names = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    mapping = {str(i): name for i, name in enumerate("ABCDEFGHI")}
    mapping.update({str(i + 9): name for i, name in enumerate("KLMNOPQRSTUVWXY")})
    mapping.update({"24": "J", "25": "Z"})
    resolved = MODULE.resolve_class_ids(names, mapping)
    assert len(resolved) == 26
    assert resolved["J"] == 24
    assert resolved["Z"] == 25


def test_class_map_must_match_manifest_classes():
    try:
        MODULE.resolve_class_ids({"A", "B"}, {"0": "A"})
    except ValueError:
        return
    raise AssertionError("incomplete class map accepted")
