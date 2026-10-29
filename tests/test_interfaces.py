"""Interface tests: file formats round-trip, Pi-safe imports, module boundaries."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rcr import io
from rcr.types import ContactPlan, Stance

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "data" / "examples"


@pytest.fixture(scope="module", autouse=True)
def contact_plan_example(tmp_path_factory):
    """No example contact plan is kept yet: a one-move plan on N1, for the format."""
    wall = io.load_wall(EXAMPLES / "N1.json")
    moved = Stance({**wall.start.contacts, "right_hand": "c29_r27"})
    path = tmp_path_factory.mktemp("plan") / "contact_plan.json"
    io.save_contact_plan(ContactPlan(wall.id, [wall.start, moved]), path)
    FILES["contact_plan.json"] = path


FILES = {name: EXAMPLES / name for name in ("N1.json", "N2.json", "N3.json", "trajectory.json")}
FORMATS = [
    ("N1.json", io.load_wall, io.save_wall),
    ("N2.json", io.load_wall, io.save_wall),
    ("N3.json", io.load_wall, io.save_wall),
    ("contact_plan.json", io.load_contact_plan, io.save_contact_plan),
    ("trajectory.json", io.load_trajectory, io.save_trajectory),
]


@pytest.mark.parametrize(("name", "load", "save"), FORMATS)
def test_example_round_trips(name, load, save, tmp_path):
    obj = load(FILES[name])
    save(obj, tmp_path / name)
    assert load(tmp_path / name) == obj


@pytest.mark.parametrize(("name", "load", "save"), FORMATS)
def test_wrong_schema_is_rejected(name, load, save, tmp_path):
    data = json.loads(FILES[name].read_text())
    data["schema"] = "something_else/1"
    (tmp_path / name).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        load(tmp_path / name)


# Microban's Pi environment [Microban pyproject.toml]: anything else must not be imported.
PI_ALLOWED = {"rcr", "numpy", "placo", "onnxruntime"}

PI_IMPORT_PROBE = """
import sys
before = set(sys.modules)
import rcr.types, rcr.io, rcr.controller, rcr.robot, rcr.robot.microban
import rcr.link, rcr.link.raspberry_pi
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


def test_the_model_comes_from_microban_model_unless_a_variable_says_otherwise(
    monkeypatch, tmp_path
):
    from rcr.robot import microban

    for name, find in (("MICROBAN_URDF", microban.urdf_path), ("MICROBAN_XML", microban.mjcf_path)):
        monkeypatch.delenv("MICROBAN_URDF", raising=False)
        monkeypatch.delenv("MICROBAN_XML", raising=False)
        try:
            default = find()  # the package's climbing model, if installed
        except FileNotFoundError as missing:
            assert "uv sync" in str(missing)  # says what to do
        else:
            assert "microban_model" in str(default) and "climb" in default.name
        other = tmp_path / f"{name}.file"
        monkeypatch.setenv(name, str(other))
        with pytest.raises(FileNotFoundError, match=name):
            find()  # it names the variable that leads nowhere
        other.write_text("")
        assert find() == other
