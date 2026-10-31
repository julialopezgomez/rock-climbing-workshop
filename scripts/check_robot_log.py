"""Compare a robot run's log with a dry run's (both from scripts/robot_policy.py): is the robot's
feedback what the policy saw in the sim, and do its corrections oscillate?

    uv run --no-sync scripts/check_robot_log.py runs/policy/<robot>.npz runs/policy/<dry run>.npz

Run both with the same policy, data and --seconds. Each line prints the robot, then the sim:
* feedback: the fields received, ticks with no new frame (Wi-Fi), q missing;
* in the first STILL_S (the first move barely started): gravity in the trunk frame from the IMU (the
  angle between the two says whether the IMU's orientation convention matches the sim's), and the
  joint velocities' spread (noise shows as a larger spread on the robot);
* tracking: the joints' error to the reference, the policy's correction (target - feedforward) and
  how often its action sits at +-1;
* oscillation: the correction's strongest frequency and amplitude, legs and arms apart.
"""

import argparse

import numpy as np

STILL_S = 1.0  # [s] at the start of the climb
N = 19  # joints


def summary(path: str) -> dict:
    d = np.load(path)
    t, q, ff, target = d["t"], d["q"], d["feedforward"], d["target"]
    joints = [str(j) for j in d["joints"]]
    still = t < STILL_S
    obs, action = d["obs"], d["action"]
    gravity = np.nanmean(obs[still, 4 * N : 4 * N + 3], axis=0)
    dq = obs[:, 3 * N : 4 * N]
    stamps = d["robot_time"]
    stale = float(np.mean(np.diff(stamps) <= 0)) if np.isfinite(stamps).all() else float("nan")
    correction = target - ff
    dt = float(np.median(np.diff(t)))
    out = {
        "fields": sorted(k[6:] for k in d.files if k.startswith("frame_")),
        "stale": stale,
        "q missing": float(np.mean(np.isnan(q))),
        "gravity": gravity / np.linalg.norm(gravity),
        "dq spread": float(np.nanmean(np.nanstd(dq[still], axis=0))),
        "error": float(np.nanmean(np.abs(q - ff))),
        "correction": float(np.nanmean(np.abs(correction))),
        "saturated": float(np.nanmean(np.abs(action) > 0.95)),
    }
    for part, words in (("legs", ("hip", "knee", "ankle")), ("arms", ("shoulder", "elbow"))):
        cols = [i for i, j in enumerate(joints) if any(w in j for w in words)]
        signal = np.nan_to_num(correction[:, cols] - np.nanmean(correction[:, cols], axis=0))
        spectrum = np.abs(np.fft.rfft(signal, axis=0)).mean(axis=1)
        freqs = np.fft.rfftfreq(len(signal), dt)
        k = 1 + int(np.argmax(spectrum[1:]))  # skip the constant
        out[f"{part} oscillation"] = (freqs[k], 2 * spectrum[k] / len(signal))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("robot", help="the robot run's log")
    parser.add_argument("sim", help="the dry run's log, same policy and data")
    args = parser.parse_args()
    robot, sim = summary(args.robot), summary(args.sim)
    print(f"fields received: robot {robot['fields']}\n                 sim   {sim['fields']}")
    for key, unit in (("stale", "of ticks"), ("q missing", "of values")):
        print(f"{key:<11}: robot {100 * robot[key]:5.1f} %  sim {100 * sim[key]:5.1f} %  {unit}")
    angle = np.degrees(np.arccos(np.clip(robot["gravity"] @ sim["gravity"], -1, 1)))
    print(f"gravity    : robot {np.round(robot['gravity'], 2)}  sim {np.round(sim['gravity'], 2)}  "
          f"apart {angle:.0f} deg (over ~10: the IMU convention differs)")  # fmt: skip
    for key, unit in (("dq spread", "rad/s"), ("error", "rad"), ("correction", "rad")):
        print(f"{key:<11}: robot {robot[key]:.3f}  sim {sim[key]:.3f}  {unit}")
    print(f"saturated  : robot {100 * robot['saturated']:.0f} %  sim {100 * sim['saturated']:.0f} %"
          "  of actions at +-1")  # fmt: skip
    for part in ("legs", "arms"):
        (fr, ar), (fs, as_) = robot[f"{part} oscillation"], sim[f"{part} oscillation"]
        print(f"{part} oscill.: robot {fr:4.1f} Hz, {ar:.3f} rad  sim {fs:4.1f} Hz, {as_:.3f} rad")


if __name__ == "__main__":
    main()
