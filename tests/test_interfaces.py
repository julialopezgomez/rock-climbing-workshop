"""Interface tests: file formats round-trip, Pi-safe imports, module boundaries."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rcr import io

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "data" / "examples"

FORMATS = [
    ("wall_hard.json", io.load_wall, io.save_wall),
    ("contact_plan_hard.json", io.load_contact_plan, io.save_contact_plan),
    ("trajectory.json", io.load_trajectory, io.save_trajectory),
]


@pytest.mark.parametrize(("name", "load", "save"), FORMATS)
def test_example_round_trips(name, load, save, tmp_path):
    obj = load(EXAMPLES / name)
    save(obj, tmp_path / name)
    assert load(tmp_path / name) == obj


@pytest.mark.parametrize(("name", "load", "save"), FORMATS)
def test_wrong_schema_is_rejected(name, load, save, tmp_path):
    data = json.loads((EXAMPLES / name).read_text())
    data["schema"] = "something_else/1"
    (tmp_path / name).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        load(tmp_path / name)


# Microban's Pi environment [Microban pyproject.toml]: anything else must not be imported.
PI_ALLOWED = {"rcr", "numpy", "placo", "onnxruntime"}

PI_IMPORT_PROBE = """
import sys
before = set(sys.modules)
import rcr.types, rcr.io, rcr.controller, rcr.link, rcr.robot
new = {m.split(".")[0] for m in set(sys.modules) - before}
print("\\n".join(sorted(new - set(sys.stdlib_module_names))))
"""


def test_robot_side_imports_are_pi_safe():
    out = subprocess.run(
        [sys.executable, "-c", PI_IMPORT_PROBE], capture_output=True, text=True, check=True
    ).stdout.split()
    assert set(out) <= PI_ALLOWED, f"not available on the Pi: {set(out) - PI_ALLOWED}"


def test_planner_never_imports_optimiser_or_controller():
    for path in (ROOT / "src" / "rcr" / "planner").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # Relative imports from rcr/planner/: level 2 ("from ..x") resolves to rcr.x.
                module = ("rcr." if node.level >= 2 else "") + (node.module or "")
                names = [module] + [f"{module}.{a.name}".lstrip(".") for a in node.names]
            else:
                continue
            for name in names:
                assert not name.startswith(("rcr.optimiser", "rcr.controller")), (
                    f"{path.name} imports {name}"
                )
