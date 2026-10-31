"""Train the residual tracking policy (rcr.controller.tracking) with PPO in the organisers' sim.

    uv run --group rl scripts/train_controller.py baseline                 # playback only (a = 0)
    uv run --group rl scripts/train_controller.py train --steps 2000000    # PPO, then ONNX export
    uv run --group rl scripts/train_controller.py evaluate                 # the ONNX policy
    uv run --group rl mjpython scripts/train_controller.py evaluate --watch   # in MuJoCo's viewer
    uv run --group rl scripts/train_controller.py curriculum --init POLICY --out DIR   # alpha up
    (--watch on baseline shows playback. macOS needs mjpython for the viewer.)

The data folder (--data, build/rl by default) holds the reference: sim_wall.json (the organisers'
Wall) and trajectory.json (ours, rcr.io, with planned_positions and trunk_poses). The moves are
read from the reference itself: where its balls sit, by forward kinematics. A teammate's .npz (one
array q: trunk x, y, z, qw, qx, qy, qz, then the 19 MICROBAN_JOINTS, a row every 0.02 s) becomes
a data folder with:

    uv run --group rl scripts/train_controller.py import wall-1-ik.npz --wall N1 --data build/rl/n1

An episode is one transition [Ours]:
* reward each step, least squares: 1 - W_Q |planned(t) - q|^2 / SIGMA_Q^2 - W_B d^2 / SIGMA_B^2,
  d the moving ball's distance to where the reference puts it at t (the sim's forward kinematics
  of planned_positions and trunk_poses): joint tracking alone misses the trunk's drift, which
  moves the ball centimetres off. Near the target it equals the earlier exp(-x) ~ 1 - x; far from
  it the squares keep their slope, where the exponential went flat. At most 1, floored at
  -REWARD_FLOOR;
* success (terminated): the moving ball hooked in its new hold, the others still in theirs; plus
  one reward per step left before the time limit, so finishing early never loses;
* failure (terminated): a fixed limb out of its hold; minus REWARD_FLOOR per step left and the
  transition's planned steps, so falling always costs more than staying, however bad;
* time limit (truncated): the transition's planned end + SLACK_S.
After a success the next transition starts where this one ended (the sim is not reset). Otherwise
a transition is drawn at random and the robot placed at its reference start (reference state
initialisation, Peng et al. 2018), settled for SETTLE_S, and kept only if all four balls are hooked.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from climbing_sim import JOINTS as SIM_JOINTS
from climbing_sim import ClimbingSim, Wall
from stable_baselines3.common.callbacks import BaseCallback

from rcr import io
from rcr.controller import tracking
from rcr.types import LIMBS, MICROBAN_JOINTS, Trajectory

HOOK_TOL = 0.006  # [m] a ball this close to a socket's centre sits in it [simulator hooked()]
NPZ_DT = 0.02  # [s] between the rows of a teammate's .npz
SIGMA_Q = 0.25  # [rad] joint error (19 joints together) whose cost is W_Q [Ours]
SIGMA_B = 0.03  # [m] moving ball's distance to the reference whose cost is W_B [Ours]
W_Q, W_B = 0.5, 0.5  # weights of the joint and ball terms [Ours]
# a step's reward stays >= -30: above the worst playback cost measured (28 on N1, 20 on the
# pre-loaded reference), so the squares keep their slope wherever playback goes [Ours]
REWARD_FLOOR = 30.0
SLACK_S = 1.0  # [s] time allowed after a transition's planned end [Ours]
SETTLE_S = 0.5  # [s] holding the reference start after a reset [Ours]
TO_SIM = [tracking.JOINTS.index(n) for n in SIM_JOINTS]  # ours -> the sim's joint order
FROM_SIM = [list(SIM_JOINTS).index(n) for n in tracking.JOINTS]  # the sim's -> ours


class TrackingEnv(gym.Env):
    def __init__(self, data: Path, alpha: float = tracking.ALPHA, kp_fw: float = tracking.KP_FW,
                 seed: int = 0):  # fmt: skip
        self.sim = ClimbingSim(Wall.load(str(data / "sim_wall.json")), cameras=[],
                               kp_fw=kp_fw, seed=seed)  # fmt: skip
        self.kp_fw = kp_fw  # the servos' P gain
        self.alpha = alpha  # the policy's share of the targets
        self.trajectory = io.load_trajectory(data / "trajectory.json")
        self.ref = tracking.reference(self.trajectory, kp_fw)
        self.low, self.high = self.sim.joint_range[FROM_SIM].T
        self.dt = self.sim.control_dt
        self.balls = reference_balls(self.sim, self.trajectory)
        self.moves = moves(self.balls, self.sim.wall.sockets(), self.trajectory.times)
        self.k, self.chain = 0, False
        n = len(tracking.JOINTS)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (tracking.OBS_SIZE,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.chain and self.k + 1 < len(self.moves):
            self.k += 1  # carry on from where the last transition ended
            obs = self.sim.observation()
        else:
            obs = self.place()
        self.chain = False
        self.t = self.moves[self.k]["t0"]
        return self.observe(obs), {"move": self.k}

    def place(self, tries: int = 20) -> dict:
        """A random transition's reference start, settled with all four balls hooked."""
        for _ in range(tries):
            self.k = int(self.np_random.integers(len(self.moves)))
            t0 = self.moves[self.k]["t0"]
            i = int(np.searchsorted(self.trajectory.times, t0))
            pose = self.trajectory.trunk_poses[i]
            names, angles = self.trajectory.joint_names, self.trajectory.planned_positions[i]
            joints = dict(zip(names, angles, strict=True))
            self.sim.reset(pose[:3], pose[3:], joints)
            ff = tracking.at(self.ref, t0)[0]
            for _ in range(round(SETTLE_S / self.dt)):
                obs = self.sim.step(ff[TO_SIM])
            if self.sim.hooked() == self.moves[self.k]["before"]:
                return obs
        raise RuntimeError(f"no reference start held after {tries} tries")

    def observe(self, obs) -> np.ndarray:
        planned = tracking.at(self.ref, self.t)[1]
        q, dq = obs["q"][FROM_SIM], obs["dq"][FROM_SIM]
        return tracking.observation(planned, q, dq, obs["quat"], obs["gyro"])

    def step(self, action):
        move = self.moves[self.k]
        self.t += self.dt
        ff, planned = tracking.at(self.ref, self.t)
        q_target = tracking.targets(ff, action, self.alpha, self.low, self.high)
        obs = self.sim.step(q_target[TO_SIM])
        error = planned - obs["q"][FROM_SIM]
        path = self.balls[:, LIMBS.index(move["limb"])]
        ref_ball = [np.interp(self.t, self.trajectory.times, path[:, c]) for c in range(3)]
        d = float(np.linalg.norm(self.sim.ball_positions()[move["limb"]] - ref_ball))
        cost = W_Q * (error @ error) / SIGMA_Q**2 + W_B * d**2 / SIGMA_B**2
        reward = max(1.0 - cost, -REWARD_FLOOR)
        hooked = self.sim.hooked()
        failed = any(hooked[limb] != move["before"][limb] for limb in move["fixed"])
        success = not failed and hooked[move["limb"]] == move["after"][move["limb"]]
        steps_left = max(0, round((move["t1"] + SLACK_S - self.t) / self.dt))
        if success:
            reward += steps_left
        if failed:
            reward -= REWARD_FLOOR * steps_left + round((move["t1"] - move["t0"]) / self.dt)
        self.chain = success
        error = float(abs(error).mean())
        info = {"move": self.k, "success": success, "failed": failed, "error": error, "ball": d}
        return self.observe(obs), float(reward), success or failed, steps_left == 0, info


def reference_balls(sim, trajectory) -> np.ndarray:
    """(samples, 4, 3): where the reference puts each ball (LIMBS order), by forward kinematics."""
    names = trajectory.joint_names
    out = []
    for angles, pose in zip(trajectory.planned_positions, trajectory.trunk_poses, strict=True):
        q = np.array([dict(zip(names, angles, strict=True))[n] for n in SIM_JOINTS])
        balls = sim.ball_positions_at(q, pose[:3], pose[3:])
        out.append([balls[limb] for limb in LIMBS])
    return np.array(out)


def moves(balls, sockets, times) -> list[dict]:
    """Per transition: its time span, the moving limb, and the socket of each ball before and
    after (indices in the sim's wall, as sim.hooked() gives them). Read from the reference: a
    stance is a sample with all four balls in sockets, and a move runs from the first sample of one
    stance to the first sample of the next."""
    stances = []  # (sample, each ball's socket in LIMBS order)
    for i, at in enumerate(balls):
        d = np.linalg.norm(sockets[None] - at[:, None], axis=2)  # (4 balls, sockets)
        if (d.min(axis=1) < HOOK_TOL).all():
            stance = tuple(int(j) for j in d.argmin(axis=1))
            if not stances or stance != stances[-1][1]:
                stances.append((i, stance))
    out = []
    for (i0, a), (i1, b) in zip(stances, stances[1:], strict=False):
        (k,) = [k for k in range(len(LIMBS)) if a[k] != b[k]]  # one limb moves
        out.append({
            "t0": times[i0], "t1": times[i1], "limb": LIMBS[k],
            "fixed": [other for other in LIMBS if other != LIMBS[k]],
            "before": dict(zip(LIMBS, a, strict=True)), "after": dict(zip(LIMBS, b, strict=True)),
        })  # fmt: skip
    return out


def import_npz(path: Path, wall: str, data: Path) -> None:
    """A teammate's .npz reference as a data folder. It is kinematic: no pre-load against sag, so
    positions = planned_positions."""
    q = np.load(path)["q"]
    rows = q[:, 7:].tolist()
    times = [round(k * NPZ_DT, 9) for k in range(len(q))]
    trajectory = Trajectory(list(MICROBAN_JOINTS), times, rows, rows, q[:, :7].tolist())
    data.mkdir(parents=True, exist_ok=True)
    Wall.load(wall).to_json(str(data / "sim_wall.json"))
    io.save_trajectory(trajectory, data / "trajectory.json")
    found = moves(
        reference_balls(ClimbingSim(wall, cameras=[]), trajectory), Wall.load(wall).sockets(), times
    )
    print(f"saved {data}: {times[-1]:.1f} s, {len(found)} moves")


def evaluate(data: Path, episodes: int, policy: Path | None = None, watch: bool = False,
             kp_fw: float = tracking.KP_FW, alpha: float | None = None) -> dict:  # fmt: skip
    """Episodes with the ONNX policy (at the alpha and servo gain it was trained with, or at the
    alpha given), or playback (alpha 0, at kp_fw) without one: what the policy has to beat. Each
    from a random reference start, or, with watch, in MuJoCo's viewer and chained (a success
    carries on to the next move) like a climb. Prints each move, returns the totals."""
    act = lambda obs: np.zeros(len(tracking.JOINTS))  # noqa: E731
    own_alpha = 0.0
    if policy is not None:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(policy))
        act = lambda obs: session.run(None, {"obs": obs[None]})[0][0]  # noqa: E731
        meta = json.loads(policy.with_suffix(".json").read_text())
        own_alpha, kp_fw = meta["alpha"], meta.get("kp_fw", tracking.KP_FW)  # older: all at 125
    env = TrackingEnv(data, own_alpha if alpha is None else alpha, kp_fw)
    env.reset(seed=0)
    if watch:
        env.sim.view()
    results = {k: [] for k in range(len(env.moves))}
    for _ in range(episodes):
        env.chain = env.chain and watch
        obs, _ = env.reset()
        k, errors, balls, actions, done, info = env.k, [], [], [], False, {}
        while not done:
            action = act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            errors.append(info["error"])
            balls.append(info["ball"])
            actions.append(np.abs(np.clip(action, -1.0, 1.0)))
            done = terminated or truncated
        end = "success" if info["success"] else "failed" if info["failed"] else "timeout"
        saturated = float(np.mean(np.array(actions) > 0.95))
        results[k].append((end, np.mean(errors), np.mean(balls), saturated))
        if watch:
            print(f"move {k} ({env.moves[k]['limb']}): {end}")
    for k, runs in results.items():
        if not runs:
            continue
        ends = [r[0] for r in runs]
        mean = np.mean(np.array([r[1:] for r in runs]), axis=0)
        print(f"move {k} ({env.moves[k]['limb']:<10}): {ends.count('success')}/{len(runs)} "
              f"success, {ends.count('failed')} failed, {ends.count('timeout')} timeout; "
              f"joint error {mean[0]:.3f} rad, ball {1000 * mean[1]:.0f} mm from the reference, "
              f"|a| > 0.95 on {100 * mean[2]:.0f} % of joint-steps")  # fmt: skip
    runs = [r for moves_runs in results.values() for r in moves_runs]
    ends = [r[0] for r in runs]
    mean = np.mean(np.array([r[1:] for r in runs]), axis=0)
    n = len(runs)
    totals = {"success": ends.count("success") / n, "failed": ends.count("failed") / n,
              "timeout": ends.count("timeout") / n, "joint_error": float(mean[0]),
              "ball_mm": float(1000 * mean[1]), "saturation": float(mean[2])}  # fmt: skip
    print("all moves: " + ", ".join(f"{k} {v:.3f}" for k, v in totals.items()))
    return totals


def train(data: Path, steps: int, envs: int, out: Path, alpha: float, kp_fw: float,
          norm_reward: bool, seed: int = 0, log_std_init: float = 0.0) -> None:  # fmt: skip
    """PPO (SB3 defaults: advantages normalised per minibatch). norm_reward: rewards scaled by a
    running estimate of the returns' std (VecNormalize), so the value function learns numbers
    near 1, not thousands; the observations are left as they are, so the ONNX export and
    Controller need nothing more. log_std_init: the log of the starting exploration noise (0: an
    std of 1 on actions in [-1, 1]; -1: 0.37)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    vec = make_vec_env(TrackingEnv, envs, seed=seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"data": data, "alpha": alpha, "kp_fw": kp_fw})  # fmt: skip
    if norm_reward:
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True)
    kwargs = {"log_std_init": log_std_init}
    model = PPO("MlpPolicy", vec, verbose=1, seed=seed, policy_kwargs=kwargs)
    history = History()
    model.learn(total_timesteps=steps, callback=history)
    save(model, history, out, TrackingEnv(data, alpha, kp_fw), norm_reward)


def save(model, history, out: Path, env: TrackingEnv, norm_reward: bool = False) -> None:
    """The model (.zip, to resume), its history and plot, and the ONNX policy with its .json."""
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(f"{out}.zip")  # out may hold dots (alpha0.1): append, never replace
    rows = history.rows
    columns = {k: np.array([r.get(k, np.nan) for r in rows]) for k in rows[0]}
    np.savez(f"{out}_history.npz", norm_reward=norm_reward, **columns)
    export(model, out, env)
    plot(Path(f"{out}_history.npz"))


def curriculum(data: Path, init: Path, out: Path, alphas: list[float], chunk_steps: int,
               max_chunks: int, gate_success: float, gate_falls: float, episodes: int,
               envs: int) -> None:  # fmt: skip
    """Raise alpha stage by stage [Ours]. Each stage resumes PPO (network, optimiser, value
    function) from the last one at the new alpha, first evaluating the last policy as it is at the
    new alpha (the handover), then trains in chunks of chunk_steps until its evaluation passes the
    gate (slot-ins >= gate_success and falls <= gate_falls, as fractions of the episodes), at most
    max_chunks times; a stage that never passes stops the curriculum. init: a trained policy (path
    without extension: its .zip, .onnx, .json), whose servo gain every stage keeps. Every
    evaluation is a row of out/curriculum.tsv; out/STATUS.md shows them all."""
    import time

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    kp_fw = json.loads(Path(f"{init}.json").read_text()).get("kp_fw", tracking.KP_FW)
    out.mkdir(parents=True, exist_ok=True)
    columns = ["time", "alpha", "phase", "steps", "success", "failed", "timeout", "joint_error",
               "ball_mm", "saturation", "gate"]  # fmt: skip
    rows = []

    def record(alpha, phase, steps, result, gate):
        rows.append([time.strftime("%H:%M"), f"{alpha:g}", phase, str(steps)]
                    + [f"{result[c]:.3f}" for c in columns[4:10]] + [gate])  # fmt: skip
        (out / "curriculum.tsv").write_text(
            "\n".join("\t".join(r) for r in [columns, *rows]) + "\n"
        )
        table = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        table += ["| " + " | ".join(r) + " |" for r in rows]
        (out / "STATUS.md").write_text(
            f"# Curriculum from {init} (kp_fw {kp_fw:g})\n\n" + "\n".join(table) + "\n"
        )

    previous = init
    for alpha in alphas:
        stage = out / f"alpha{alpha:g}"
        handover = evaluate(data, episodes, Path(f"{previous}.onnx"), alpha=alpha)
        record(alpha, "handover", 0, handover, "")
        vec = make_vec_env(TrackingEnv, envs, seed=0, vec_env_cls=SubprocVecEnv,
                           env_kwargs={"data": data, "alpha": alpha, "kp_fw": kp_fw})  # fmt: skip
        model = PPO.load(f"{previous}.zip", env=vec, device="cpu")
        history, passed = History(), False
        for chunk in range(1, max_chunks + 1):
            model.learn(total_timesteps=chunk_steps, callback=history, reset_num_timesteps=False)
            save(model, history, stage, TrackingEnv(data, alpha, kp_fw))
            result = evaluate(data, episodes, Path(f"{stage}.onnx"))
            passed = result["success"] >= gate_success and result["failed"] <= gate_falls
            record(
                alpha, f"chunk {chunk}", model.num_timesteps, result, "pass" if passed else "fail"
            )
            if passed:
                break
        vec.close()
        if not passed:
            print(f"alpha {alpha:g} did not pass its gate in {max_chunks} chunks: stopping")
            return
        previous = stage
    print(f"curriculum done: {out / 'STATUS.md'}")


class History(BaseCallback):
    """Per PPO rollout: the step rewards, the advantages before PPO normalises them, the value
    targets (returns), the episodes' outcomes, and the previous update's losses. Kept across
    learn() calls (a curriculum stage trains in chunks)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows, self.ends = [], []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"], strict=True):
            if done:
                self.ends.append(
                    "success" if info["success"] else "failed" if info["failed"] else "timeout"
                )
        return True

    def _on_rollout_end(self) -> None:
        buffer, logged = self.model.rollout_buffer, self.model.logger.name_to_value
        r, adv, ret = (x.ravel() for x in (buffer.rewards, buffer.advantages, buffer.returns))
        episodes = [e["r"] for e in self.model.ep_info_buffer]
        n = max(len(self.ends), 1)
        self.rows.append({
            "steps": self.num_timesteps,
            **spread("reward", r), **spread("return", ret),
            "adv_std": adv.std(), "adv_absmax": np.abs(adv).max(),
            "episode_return": np.mean(episodes) if episodes else np.nan,
            "success": self.ends.count("success") / n, "failed": self.ends.count("failed") / n,
            "timeout": self.ends.count("timeout") / n,
            "value_loss": logged.get("train/value_loss", np.nan),
            "explained_variance": logged.get("train/explained_variance", np.nan),
            "std": logged.get("train/std", np.nan),
        })  # fmt: skip
        self.ends = []


def spread(name: str, x: np.ndarray) -> dict:
    """Mean and 5th / 95th percentiles of x, keyed by name."""
    p5, p95 = np.percentile(x, (5, 95))
    return {f"{name}_mean": x.mean(), f"{name}_p5": p5, f"{name}_p95": p95}


def plot(history: Path) -> None:
    """Training curves, one chart per quantity, saved next to the history as a .png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.load(history)
    x = h["steps"] / 1e6
    # with --norm-reward, PPO's rewards (so advantages, returns, value loss) are the scaled ones
    unit = ", scaled" if "norm_reward" in h.files and h["norm_reward"] else ""
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"  # the dataviz reference palette
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    plt.rcParams.update({"font.size": 9, "text.color": ink, "axes.edgecolor": muted})
    plt.rcParams.update({"axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), facecolor=surface)
    charts = [
        ("Episode return", [("mean", h["episode_return"], blue)], None, None),
        ("Episode outcomes (fraction)", [("slot-in", h["success"], blue),
         ("fall", h["failed"], orange), ("timeout", h["timeout"], aqua)], None, None),
        (f"Step reward (mean, 5-95 %{unit})", [("mean", h["reward_mean"], blue)],
         (h["reward_p5"], h["reward_p95"]), None),
        (f"Raw advantage (before PPO normalises it{unit})", [("std", h["adv_std"], blue),
         ("max |A|", h["adv_absmax"], orange)], None, "log"),
        (f"Value targets: returns (mean, 5-95 %{unit})", [("mean", h["return_mean"], blue)],
         (h["return_p5"], h["return_p95"]), None),
        (f"Value loss{unit}", [("loss", h["value_loss"], blue)], None, "log"),
        ("Explained variance", [("explained", h["explained_variance"], blue)], None, None),
        ("Policy std (exploration)", [("std", h["std"], blue)], None, None),
    ]  # fmt: skip
    for ax, (title, series, band, scale) in zip(axes.ravel(), charts, strict=True):
        ax.set_facecolor(surface)
        if band is not None:
            ax.fill_between(x, band[0], band[1], color=series[0][2], alpha=0.18, linewidth=0)
        for label, y, color in series:
            ax.plot(x, y, color=color, linewidth=1.5, label=label)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=8)
        if scale:
            ax.set_yscale(scale)
        else:
            ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.set_xlabel("training steps (millions)")
        ax.grid(color="#e4e3df", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(history.with_suffix(".png"), dpi=130, facecolor=surface)
    print(f"saved {history.with_suffix('.png')}")


def export(model, out: Path, env: TrackingEnv) -> None:
    """The policy's mean action as ONNX (input "obs"), and what Controller checks next to it."""
    import torch

    class Mean(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            return self.policy(obs, deterministic=True)[0]

    obs = torch.zeros(1, tracking.OBS_SIZE)
    torch.onnx.export(Mean(model.policy), obs, f"{out}.onnx", input_names=["obs"],
                      dynamo=False)  # fmt: skip
    meta = {"alpha": env.alpha, "kp_fw": env.kp_fw, "joints": list(tracking.JOINTS),
            "low": env.low.tolist(), "high": env.high.tolist()}  # fmt: skip
    Path(f"{out}.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {out}.onnx and {out}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "command", choices=["import", "baseline", "evaluate", "train", "plot", "curriculum"]
    )
    parser.add_argument("npz", type=Path, nargs="?", help="import: the teammate's .npz")
    parser.add_argument("--wall", default="N1", help="import: the organisers' wall it is on")
    parser.add_argument("--data", type=Path, default=Path("build/rl"))
    parser.add_argument("--episodes", type=int, default=20, help="evaluation episodes")
    parser.add_argument("--policy", type=Path, default=Path("build/rl/policy.onnx"))
    parser.add_argument("--watch", action="store_true", help="in MuJoCo's viewer (macOS: mjpython)")
    parser.add_argument("--steps", type=int, default=2_000_000, help="training steps")
    parser.add_argument("--envs", type=int, default=8, help="parallel sims")
    parser.add_argument("--out", type=Path, default=Path("build/rl/policy"))
    parser.add_argument("--alpha", type=float, default=tracking.ALPHA, help="the policy's share")
    parser.add_argument("--norm-reward", action="store_true", help="train: VecNormalize rewards")
    parser.add_argument("--seed", type=int, default=0, help="train: PPO and environment seed")
    parser.add_argument("--kp", type=float, default=tracking.KP_FW, help="servo P gain (kp_fw)")
    parser.add_argument("--log-std-init", type=float, default=0.0, help="train: log starting noise")
    parser.add_argument("--init", type=Path, help="curriculum: the alpha 0.1 policy, no extension")
    parser.add_argument("--alphas", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--chunk-steps", type=int, default=1_000_000, help="curriculum")
    parser.add_argument("--max-chunks", type=int, default=3, help="curriculum: per stage")
    parser.add_argument("--gate-success", type=float, default=0.5, help="curriculum: fraction")
    parser.add_argument("--gate-falls", type=float, default=0.2, help="curriculum: fraction")
    args = parser.parse_args()
    if args.command == "import":
        import_npz(args.npz, args.wall, args.data)
    elif args.command == "train":
        train(
            args.data,
            args.steps,
            args.envs,
            args.out,
            args.alpha,
            args.kp,
            args.norm_reward,
            args.seed,
            args.log_std_init,
        )
    elif args.command == "curriculum":
        alphas = [float(a) for a in args.alphas.split(",")]
        curriculum(args.data, args.init, args.out, alphas, args.chunk_steps, args.max_chunks,
                   args.gate_success, args.gate_falls, args.episodes, args.envs)  # fmt: skip
    elif args.command == "plot":
        plot(Path(f"{args.out}_history.npz"))
    else:
        policy = args.policy if args.command == "evaluate" else None
        evaluate(args.data, args.episodes, policy, args.watch, args.kp)
