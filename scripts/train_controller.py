"""Train the residual tracking policy (rcr.controller.tracking) with PPO in the organisers' sim.

    uv run --group rl scripts/train_controller.py baseline                 # playback only (a = 0)
    uv run --group rl scripts/train_controller.py train --steps 2000000    # PPO, then ONNX export
    uv run --group rl scripts/train_controller.py evaluate                 # the ONNX policy
    uv run --group rl mjpython scripts/train_controller.py evaluate --watch   # in MuJoCo's viewer
    (--watch on baseline shows playback. macOS needs mjpython for the viewer.)

The data folder (--data, build/rl by default) holds the reference: sim_wall.json (the organisers'
Wall) and trajectory.json (ours, rcr.io, with planned_positions and trunk_poses). The moves are
read from the reference itself: where its balls sit, by forward kinematics. A teammate's .npz (one
array q: trunk x, y, z, qw, qx, qy, qz, then the 19 MICROBAN_JOINTS, a row every 0.02 s) becomes
a data folder with:

    uv run --group rl scripts/train_controller.py import wall-1-ik.npz --wall N1 --data build/rl/n1

An episode is one transition [Ours]:
* reward each step, bounded and positive (DeepMimic, Peng et al. 2018):
  W_Q exp(-|planned(t) - q|^2 / SIGMA_Q^2) + W_B exp(-d^2 / SIGMA_B^2) + W_ALIVE
  - W_RATE mean((a_t - a_{t-1})^2), d the moving ball's distance to where the reference puts it at
  t (the sim's forward kinematics of planned_positions and trunk_poses). At most 1;
* success (terminated): the moving ball hooked in its new hold, the others still in theirs; plus
  one reward per step left before the time limit, so finishing early never loses;
* failure (terminated): a fixed limb out of its hold; no penalty, it forgoes the steps left;
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
# Reward [Ours, after DeepMimic, Peng et al. 2018]: bounded positive terms, so staying on the wall
# pays and a fall costs the rewards it forgoes; tracking is loose, as the reference itself fails
# on some moves (a held limb pops out even when its joints are tracked within 0.01 rad)
SIGMA_Q = 0.25  # [rad] joint error (19 joints together) at which its term is exp(-1)
SIGMA_B = 0.03  # [m] moving ball's distance to the reference at which its term is exp(-1)
W_Q, W_B, W_ALIVE = 0.3, 0.3, 0.4  # joints, moving ball, every step still on the wall
W_RATE = 0.1  # times the mean square change of the action between steps (smoothness)
SLACK_S = 1.0  # [s] time allowed after a transition's planned end [Ours]
SETTLE_S = 0.5  # [s] holding the reference start after a reset [Ours]
TO_SIM = [tracking.JOINTS.index(n) for n in SIM_JOINTS]  # ours -> the sim's joint order
FROM_SIM = [list(SIM_JOINTS).index(n) for n in tracking.JOINTS]  # the sim's -> ours


class TrackingEnv(gym.Env):
    def __init__(self, data: Path, alpha: float = tracking.ALPHA, kp_fw: float = tracking.KP_FW,
                 seed: int = 0, only: list[int] | None = None):  # fmt: skip
        self.sim = ClimbingSim(Wall.load(str(data / "sim_wall.json")), cameras=[],
                               kp_fw=kp_fw, seed=seed)  # fmt: skip
        self.kp_fw = kp_fw  # the servos' P gain
        self.alpha = alpha  # the largest residual correction (rad)
        self.trajectory = io.load_trajectory(data / "trajectory.json")
        self.ref = tracking.reference(self.trajectory, kp_fw)
        self.low, self.high = self.sim.joint_range[FROM_SIM].T
        self.dt = self.sim.control_dt
        self.balls = reference_balls(self.sim, self.trajectory)
        self.moves = moves(self.balls, self.sim.wall.sockets(), self.trajectory.times)
        self.only = only  # train on these moves only (None: all)
        self.k, self.chain = 0, False
        n = len(tracking.JOINTS)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (tracking.OBS_SIZE,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        nxt = self.k + 1
        if self.chain and nxt < len(self.moves) and (not self.only or nxt in self.only):
            self.k += 1  # carry on from where the last transition ended
            obs = self.sim.observation()
        else:
            obs = self.place()
        self.chain = False
        self.t = self.moves[self.k]["t0"]
        self.last_action = np.zeros(len(tracking.JOINTS))
        return self.observe(obs), {"move": self.k}

    def place(self, tries: int = 20) -> dict:
        """A random transition's reference start, settled with all four balls hooked."""
        for _ in range(tries):
            pool = self.only or range(len(self.moves))
            self.k = int(pool[self.np_random.integers(len(pool))])
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
        ahead = tracking.at(self.ref, self.t + tracking.LOOKAHEAD_S)[1]
        q, dq = obs["q"][FROM_SIM], obs["dq"][FROM_SIM]
        return tracking.observation(planned, ahead, q, dq, obs["quat"], obs["gyro"])

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
        action = np.clip(action, -1.0, 1.0)
        rate = float(np.mean((action - self.last_action) ** 2))
        self.last_action = action
        reward = (W_Q * np.exp(-(error @ error) / SIGMA_Q**2) + W_B * np.exp(-(d**2) / SIGMA_B**2)
                  + W_ALIVE - W_RATE * rate)  # fmt: skip
        hooked = self.sim.hooked()
        failed = any(hooked[limb] != move["before"][limb] for limb in move["fixed"])
        success = not failed and hooked[move["limb"]] == move["after"][move["limb"]]
        steps_left = max(0, round((move["t1"] + SLACK_S - self.t) / self.dt))
        if success:
            reward += steps_left  # the most the steps left could earn: finishing early never loses
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
             kp_fw: float = tracking.KP_FW) -> None:  # fmt: skip
    """Episodes with the ONNX policy (at the alpha and servo gain it was trained with), or
    playback (alpha 0, at kp_fw) without one: what the policy has to beat. Each from a random
    reference start, or, with watch, in MuJoCo's viewer and chained (a success carries on to the
    next move) like a climb."""
    act = lambda obs: np.zeros(len(tracking.JOINTS))  # noqa: E731
    alpha = 0.0
    if policy is not None:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(policy))
        act = lambda obs: session.run(None, {"obs": obs[None]})[0][0]  # noqa: E731
        meta = json.loads(policy.with_suffix(".json").read_text())
        alpha, kp_fw = meta["alpha"], meta.get("kp_fw", tracking.KP_FW)  # older: all at 125
    env = TrackingEnv(data, alpha, kp_fw)
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


def train(data: Path, steps: int, envs: int, out: Path, alpha: float, kp_fw: float,
          norm_reward: bool, seed: int = 0, log_std_init: float = 0.0,
          only: list[int] | None = None) -> None:  # fmt: skip
    """PPO (SB3 defaults: advantages normalised per minibatch). norm_reward: rewards scaled by a
    running estimate of the returns' std (VecNormalize), so the value function learns numbers
    near 1, not thousands; the observations are left as they are, so the ONNX export and
    Controller need nothing more. log_std_init: the log of the starting exploration noise (0: an
    std of 1 on actions in [-1, 1]; -1: 0.37)."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    kwargs = {"data": data, "alpha": alpha, "kp_fw": kp_fw, "only": only}
    vec = make_vec_env(TrackingEnv, envs, seed=seed, vec_env_cls=SubprocVecEnv, env_kwargs=kwargs)
    if norm_reward:
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True)
    model = PPO("MlpPolicy", vec, verbose=1, seed=seed,
                policy_kwargs={"log_std_init": log_std_init})  # fmt: skip
    with torch.no_grad():  # the residual starts at 0: training starts from playback [Silver 2018]
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    history = History(out, TrackingEnv(data, alpha, kp_fw))
    model.learn(total_timesteps=steps, callback=history)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(f"{out}.zip")  # out may hold dots (alpha0.1): append, never replace
    rows = history.rows
    columns = {k: np.array([r.get(k, np.nan) for r in rows]) for k in rows[0]}
    np.savez(f"{out}_history.npz", norm_reward=norm_reward, **columns)
    export(model, out, TrackingEnv(data, alpha, kp_fw))
    plot(Path(f"{out}_history.npz"))


class History(BaseCallback):
    """Per PPO rollout: the step rewards, the advantages before PPO normalises them, the value
    targets (returns), the episodes' outcomes, and the previous update's losses. The outcomes and
    the tracking errors also go to the printed table (climb/), and every `every` steps the policy
    is exported as <out>_<k>k.onnx, so a run can be evaluated before it ends or if it is cut."""

    def __init__(self, out: Path | None = None, env: "TrackingEnv | None" = None,
                 every: int = 100_000) -> None:  # fmt: skip
        super().__init__()
        self.out, self.env, self.every = out, env, every

    def _on_training_start(self) -> None:
        self.rows, self.ends, self.errors, self.balls = [], [], [], []
        self.next_save = self.every

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"], strict=True):
            self.errors.append(info["error"])
            self.balls.append(info["ball"])
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
        for key in ("success", "failed", "timeout"):
            self.logger.record(f"climb/{key}", round(self.rows[-1][key], 3))
        self.logger.record("climb/episodes", len(self.ends))
        self.logger.record("climb/joint_err", round(float(np.mean(self.errors)), 4))
        self.logger.record("climb/ball_mm", round(1000 * float(np.mean(self.balls)), 1))
        self.ends, self.errors, self.balls = [], [], []
        if self.out is not None and self.num_timesteps >= self.next_save:
            export(self.model, Path(f"{self.out}_{self.next_save // 1000}k"), self.env)
            self.next_save += self.every


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
    meta = {"action": tracking.ACTION, "alpha": env.alpha, "kp_fw": env.kp_fw,
            "joints": list(tracking.JOINTS),
            "low": env.low.tolist(), "high": env.high.tolist()}  # fmt: skip
    Path(f"{out}.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {out}.onnx and {out}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["import", "baseline", "evaluate", "train", "plot"])
    parser.add_argument("npz", type=Path, nargs="?", help="import: the teammate's .npz")
    parser.add_argument("--wall", default="N1", help="import: the organisers' wall it is on")
    parser.add_argument("--data", type=Path, default=Path("build/rl"))
    parser.add_argument("--episodes", type=int, default=20, help="evaluation episodes")
    parser.add_argument("--policy", type=Path, default=Path("build/rl/policy.onnx"))
    parser.add_argument("--watch", action="store_true", help="in MuJoCo's viewer (macOS: mjpython)")
    parser.add_argument("--steps", type=int, default=2_000_000, help="training steps")
    parser.add_argument("--envs", type=int, default=8, help="parallel sims")
    parser.add_argument("--out", type=Path, default=Path("build/rl/policy"))
    parser.add_argument("--alpha", type=float, default=tracking.ALPHA, help="max residual, rad")
    parser.add_argument("--norm-reward", action="store_true", help="train: VecNormalize rewards")
    parser.add_argument("--seed", type=int, default=0, help="train: PPO and environment seed")
    parser.add_argument("--kp", type=float, default=tracking.KP_FW, help="servo P gain (kp_fw)")
    parser.add_argument("--log-std-init", type=float, default=0.0, help="train: log starting noise")
    parser.add_argument("--moves", help="train: only these moves, e.g. 6,18,26 (default: all)")
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
            [int(k) for k in args.moves.split(",")] if args.moves else None,
        )
    elif args.command == "plot":
        plot(Path(f"{args.out}_history.npz"))
    else:
        policy = args.policy if args.command == "evaluate" else None
        evaluate(args.data, args.episodes, policy, args.watch, args.kp)
