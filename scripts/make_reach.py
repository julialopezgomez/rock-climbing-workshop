"""Sample each limb's reach and write data/robot/reach.json, the planner's reach sets (reach/1).

uv sync --group optimiser
export MICROBAN_URDF=/path/to/microban/src/model/urdf/robot.urdf
uv run scripts/make_reach.py                 # 20000 samples per limb
uv run scripts/make_reach.py --samples 5000 --out /tmp/reach.json

For each limb, as SL1M samples its effector workspaces [Tonneau et al. 2020, §V-A.2]:
  - random angles for the limb's joints, uniform within the URDF's limits;
  - a trunk orientation drawn from those the optimiser used on the example climb
    (data/examples/trajectory.json, trunk_poses), so the set covers how the trunk leans [Ours];
  - the ball centre (rcr.robot.hooks) minus the trunk position, in the wall frame.
The set K_limb is kept as its support function along DIRECTIONS shared by all limbs (outer
approximation of the samples' convex hull, plus MARGIN): {o : n_k . o <= h_k}. Sharing the
directions makes a stance's trunk region one small LP, whatever the number of limbs (see
rcr.planner.reach). It is kinematics only: no torque, no collisions [Ours].
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rcr import io
from rcr.optimiser import robot_model as rm
from rcr.optimiser.transition import TransitionConfig
from rcr.robot.hooks import HOOKS, ball_centre
from rcr.types import LIMBS, MICROBAN_JOINTS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "robot" / "reach.json"
N_DIRECTIONS = 100  # directions on a hemisphere, used with their opposites [Ours]
MARGIN = 0.005  # [m] added to every support value: the samples miss the hull's edges [Ours]
# [m] the trunk's distance out from the wall: the example climb's (0.083 .. 0.174 m), widened
TRUNK_Y = (-0.20, -0.06)


def directions(n: int) -> np.ndarray:
    """2n + 6 unit vectors: n over a hemisphere (Fibonacci), their opposites, and the axes."""
    k = np.arange(n) + 0.5
    z = k / n  # the upper hemisphere
    r = np.sqrt(1 - z**2)
    phi = np.pi * (1 + 5**0.5) * k
    half = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    axes = np.eye(3)
    return np.concatenate([half, -half, axes, -axes])


def chain(urdf, limb: str) -> list:
    """The joints from the trunk to the limb's hook, trunk first."""
    by_child = {j.child: j for j in urdf.joints}
    joints, link = [], HOOKS[limb].link
    while link != rm.ROOT_LINK:
        joints.append(by_child[link])
        link = joints[-1].parent
    return joints[::-1]


def rotation(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def sample(urdf, limb: str, quats: np.ndarray, n: int, rng) -> np.ndarray:
    """[m] n ball - trunk offsets of the limb, in the wall frame. Forward kinematics along the
    limb's chain, as robot_model.link_poses (which works in CasADi types, too slow here)."""
    joints = chain(urdf, limb)
    ball = ball_centre(HOOKS[limb])
    out = np.empty((n, 3))
    for s in range(n):
        r, p = rotation(quats[rng.integers(len(quats))]), np.zeros(3)
        for j in joints:
            p = p + r @ np.asarray(j.xyz, dtype=float)
            r = r @ np.asarray(j.rotation, dtype=float)
            if j.revolute:
                r = r @ rm._rodrigues(
                    np.asarray(j.axis, dtype=float), rng.uniform(j.lower, j.upper)
                )
        out[s] = p + r @ ball
    return out


def check_against_link_poses(urdf, limb: str, quat: np.ndarray) -> None:
    """The numpy chain above gives robot_model.link_poses' ball for one random pose."""
    rng = np.random.default_rng(1)
    q = np.zeros(rm.NQ)
    q[3:7] = quat
    index = {name: 7 + i for i, name in enumerate(MICROBAN_JOINTS)}
    joints = chain(urdf, limb)
    r, p = rotation(quat), np.zeros(3)
    for j in joints:
        p = p + r @ np.asarray(j.xyz, dtype=float)
        r = r @ np.asarray(j.rotation, dtype=float)
        if j.revolute:
            q[index[j.name]] = rng.uniform(j.lower, j.upper)
            r = r @ rm._rodrigues(np.asarray(j.axis, dtype=float), q[index[j.name]])
    position, rot = rm.link_poses(urdf, q)[HOOKS[limb].link]
    expected = np.asarray(position).ravel() + np.asarray(rot) @ ball_centre(HOOKS[limb])
    assert np.allclose(p + r @ ball_centre(HOOKS[limb]), expected, atol=1e-9), limb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=20000, help="per limb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    urdf = rm.read_urdf(rm.urdf_path())
    trajectory = io.load_trajectory(ROOT / "data" / "examples" / "trajectory.json")
    quats = np.array(trajectory.trunk_poses)[:, 3:7]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    n = directions(N_DIRECTIONS)
    bound = rm.limb_reach()
    limbs = {}
    for limb in LIMBS:
        check_against_link_poses(urdf, limb, quats[0])
        offsets = sample(urdf, limb, quats, args.samples, rng)
        support = (offsets @ n.T).max(axis=0) + MARGIN
        limbs[limb] = [round(float(v), 5) for v in support]
        _, reach = bound[limb]
        print(
            f"{limb:<11} ball up to {np.linalg.norm(offsets, axis=1).max():.3f} m from the trunk; "
            f"x {offsets[:, 0].min():+.3f}..{offsets[:, 0].max():+.3f}, "
            f"z {offsets[:, 2].min():+.3f}..{offsets[:, 2].max():+.3f} m; "
            f"bound from its first joint {reach:.3f} m (robot_model.limb_reach)"
        )
    data = {
        "schema": "reach/1",
        "source": {
            "microban_commit": rm.CHECKED_COMMIT,
            "urdf_sha256": hashlib.sha256(rm.urdf_path().read_bytes()).hexdigest(),
            "samples_per_limb": args.samples,
            "trunk_orientations": len(quats),
            "seed": args.seed,
        },
        "margin": MARGIN,
        "contact_offset": list(TransitionConfig().contact_offset),
        "trunk_y": list(TRUNK_Y),
        "directions": [[round(float(v), 6) for v in row] for row in n],
        "support": limbs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
