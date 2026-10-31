
"""Diagnostics for comparing pseudoinverse and QP inverse kinematics.

Both methods are expected to use the same task-space trajectory,
50 Hz sampling, and initial configuration.

Plots:
    1. Maximum joint displacement per sample
    2. Maximum joint velocity / URDF velocity limit
    3. Maximum absolute joint acceleration
    4. Minimum normalized clearance from joint position limits
    5. Moving limb Cartesian tracking error (optional)
    6. Supporting limb contact error (optional)

Usage:
    plot_comparison(
        wall,
        plan,
        {"Pseudoinverse": qs_pinv, "QP": qs_qp},
        filename="build/ik_comparison.png",
    )

Or, without a wall or plan:
    plot_joint_diagnostics(
        {"Pseudoinverse": qs_pinv, "QP": qs_qp}
    )
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from rcr.kinematics import ik


# ---------------------------------------------------------------------
# Joint diagnostics
# ---------------------------------------------------------------------

def joint_diagnostics(qs):
    """Compute joint motion diagnostics from Pinocchio configurations.

    qs: array of shape (N, model.nq).

    The first 6 velocity coordinates describe the floating base.
    The remaining coordinates describe the 19 actuated joints.
    """
    qs = np.asarray(qs, dtype=float)

    if qs.ndim != 2 or qs.shape[1] != ik.model.nq:
        raise ValueError(
            f"Expected configurations of shape (N, {ik.model.nq})"
        )

    if len(qs) < 3:
        raise ValueError("At least three configurations are required")

    if not np.all(np.isfinite(qs)):
        raise ValueError("Trajectory contains non-finite configurations")

    dt = ik.DT
    q = qs[:, 7:]

    # Use Pinocchio's difference for the floating-base configuration.
    velocities = np.array([
        pin.difference(ik.model, q0, q1)[6:] / dt
        for q0, q1 in zip(qs[:-1], qs[1:])
    ])

    displacements = velocities * dt

    accelerations = np.diff(velocities, axis=0) / dt

    # Limits in Pinocchio joint order.
    velocity_limits = ik.model.velocityLimit[6:]
    lower = ik.model.lowerPositionLimit[7:]
    upper = ik.model.upperPositionLimit[7:]

    valid_v = np.isfinite(velocity_limits) & (velocity_limits > 0)

    velocity_ratio = np.full_like(velocities, np.nan)
    velocity_ratio[:, valid_v] = (
        np.abs(velocities[:, valid_v]) / velocity_limits[valid_v]
    )

    # Clearance relative to each joint's total allowed range.
    ranges = upper - lower
    valid_q = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & (ranges > 0)
    )

    clearance = np.full_like(q, np.nan)
    clearance[:, valid_q] = (
        np.minimum(
            q[:, valid_q] - lower[valid_q],
            upper[valid_q] - q[:, valid_q],
        )
        / ranges[valid_q]
    )

    times = np.arange(len(qs)) * dt

    return {
        "times": times,
        "positions": q,
        "displacements": displacements,
        "velocities": velocities,
        "accelerations": accelerations,
        "velocity_ratio": velocity_ratio,
        "clearance": clearance,
        "max_displacement": np.max(
            np.abs(displacements), axis=1
        ),
        "max_velocity_ratio": np.nanmax(
            velocity_ratio, axis=1
        ) if valid_v.any() else np.full(len(velocities), np.nan),
        "max_acceleration": np.max(
            np.abs(accelerations), axis=1
        ),
        "min_clearance": np.nanmin(
            clearance, axis=1
        ) if valid_q.any() else np.full(len(q), np.nan),
    }


# ---------------------------------------------------------------------
# Cartesian tracking diagnostics
# ---------------------------------------------------------------------

def tracking_diagnostics(wall, plan, qs):
    """Compare achieved ball positions against desired trajectories."""
    qs = np.asarray(qs)

    n_moves = len(plan.stances) - 1
    samples_per_move = round(ik.MOVE_TIME / ik.DT)

    expected = n_moves * samples_per_move + 1

    if len(qs) != expected:
        raise ValueError(
            f"Expected {expected} configurations for this plan, "
            f"received {len(qs)}"
        )

    # Compute actual ball positions for every configuration.
    actual = np.array([
        ik.ball_positions(q)
        for q in qs
    ])

    moving_errors = []
    held_errors = []
    times = []

    for move_index, (a, b) in enumerate(
        zip(plan.stances[:-1], plan.stances[1:])
    ):
        moving = [
            limb for limb in ik.LIMBS
            if a.contacts[limb] != b.contacts[limb]
        ]

        if len(moving) != 1:
            raise ValueError("Each move must change exactly one limb")

        moving_idx = ik.LIMBS.index(moving[0])

        p_a = ik.stance_targets(wall, ik.stance_key(a))
        p_b = ik.stance_targets(wall, ik.stance_key(b))

        B = ik.limb_traj(
            p_a[moving_idx],
            p_b[moving_idx],
        )

        # Include the first sample only for the first movement.
        start = 0 if move_index == 0 else 1

        for k in range(start, samples_per_move + 1):
            index = move_index * samples_per_move + k
            t = k * ik.DT

            desired = p_a.copy()
            desired[moving_idx] = B(t)

            errors = np.linalg.norm(
                actual[index] - desired,
                axis=1,
            )

            held_idx = [
                j for j in range(len(ik.LIMBS))
                if j != moving_idx
            ]

            moving_errors.append(errors[moving_idx])
            held_errors.append(np.max(errors[held_idx]))
            times.append(index * ik.DT)

    return {
        "times": np.asarray(times),
        "moving_error": np.asarray(moving_errors),
        "held_error": np.asarray(held_errors),
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_comparison(
    wall,
    plan,
    trajectories,
    filename="build/ik_comparison.png",
    show=False,
):
    """Plot and compare IK trajectories.

    trajectories: dict mapping method name -> configuration array.
    If wall or plan is None, show only joint diagnostics.
    """
    if not trajectories:
        raise ValueError("No trajectories supplied")

    with_tracking = wall is not None and plan is not None

    joint_results = {
        name: joint_diagnostics(qs)
        for name, qs in trajectories.items()
    }

    tracking_results = {}

    if with_tracking:
        tracking_results = {
            name: tracking_diagnostics(wall, plan, qs)
            for name, qs in trajectories.items()
        }

    if with_tracking:
        fig, axes = plt.subplots(
            3, 2, figsize=(15, 12),
            constrained_layout=True,
        )
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(
            2, 2, figsize=(14, 9),
            constrained_layout=True,
        )
        axes = axes.ravel()

    fig.suptitle(
        "Inverse Kinematics Trajectory Diagnostics",
        fontsize=16,
    )

    for name, result in joint_results.items():
        t = result["times"]

        # 1. Joint displacement
        axes[0].plot(
            t[1:],
            result["max_displacement"],
            label=name,
        )

        # 2. Fraction of velocity limit
        axes[1].plot(
            t[1:],
            result["max_velocity_ratio"],
            label=name,
        )

        # 3. Acceleration
        axes[2].plot(
            t[1:-1],
            result["max_acceleration"],
            label=name,
        )

        # 4. Joint-limit clearance
        axes[3].plot(
            t,
            result["min_clearance"],
            label=name,
        )

    axes[0].set_title("Maximum joint change per sample")
    axes[0].set_ylabel("Maximum |Δq| (rad)")

    axes[1].set_title("Joint velocity limit utilisation")
    axes[1].set_ylabel("Maximum |v| / velocity limit")
    axes[1].axhline(
        1.0, color="red", linestyle="--",
        label="URDF limit",
    )

    axes[2].set_title("Joint acceleration (smoothness)")
    axes[2].set_ylabel("Maximum |acceleration| (rad/s²)")

    axes[3].set_title("Joint position limit clearance")
    axes[3].set_ylabel("Minimum normalized clearance")
    axes[3].axhline(
        0.0, color="red", linestyle="--",
        label="Joint limit",
    )

    if with_tracking:
        for name, result in tracking_results.items():
            t = result["times"]

            axes[4].plot(
                t,
                result["moving_error"] * 1000,
                label=name,
            )

            axes[5].plot(
                t,
                result["held_error"] * 1000,
                label=name,
            )

        axes[4].set_title("Moving limb tracking error")
        axes[4].set_ylabel("Position error (mm)")

        axes[5].set_title("Supporting limb contact drift")
        axes[5].set_ylabel("Maximum position error (mm)")

        for ax in axes[4:]:
            ax.axhline(
                ik.EPSILON * 1000,
                color="red",
                linestyle="--",
                label="IK tolerance",
            )

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        print(f"Saved diagnostic plot to {path}")

    print_summary(joint_results, tracking_results)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "joints": joint_results,
        "tracking": tracking_results,
    }


def plot_joint_diagnostics(
    trajectories,
    filename="build/ik_joint_diagnostics.png",
    show=False,
):
    """Plot joint diagnostics without needing a wall or contact plan."""
    return plot_comparison(
        None,
        None,
        trajectories,
        filename=filename,
        show=show,
    )


# ---------------------------------------------------------------------
# Numerical summary
# ---------------------------------------------------------------------

def print_summary(joint_results, tracking_results=None):
    """Print peak motion metrics for each IK method."""
    tracking_results = tracking_results or {}

    print("\n" + "=" * 65)
    print("IK TRAJECTORY COMPARISON")
    print("=" * 65)

    for name, result in joint_results.items():
        print(f"\n{name}")
        print("-" * 45)

        max_v_ratio = result["max_velocity_ratio"]
        clearance = result["min_clearance"]

        print(
            "Max joint change:       "
            f"{np.max(result['max_displacement']):.4f} rad"
        )

        if np.any(np.isfinite(max_v_ratio)):
            peak = np.nanmax(max_v_ratio)
            print(f"Peak velocity / limit:  {peak:.3f}")
            print(
                "Velocity limits:        "
                f"{'PASS' if peak <= 1.0 + 1e-8 else 'FAIL'}"
            )
        else:
            print("Velocity limits:        Not available")

        print(
            "Peak joint acceleration:"
            f" {np.max(result['max_acceleration']):.3f} rad/s²"
        )

        if np.any(np.isfinite(clearance)):
            margin = np.nanmin(clearance)
            print(f"Minimum joint margin:   {margin:.4f}")
            print(
                "Position limits:        "
                f"{'PASS' if margin >= -1e-8 else 'FAIL'}"
            )

        if name in tracking_results:
            tracking = tracking_results[name]

            print(
                "Peak moving ball error: "
                f"{np.max(tracking['moving_error']) * 1000:.3f} mm"
            )
            print(
                "Peak held ball error:   "
                f"{np.max(tracking['held_error']) * 1000:.3f} mm"
            )

    print("=" * 65)


# ---------------------------------------------------------------------
# Command-line entry point (saved .npy trajectories)
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot joint diagnostics from saved IK trajectories"
    )

    parser.add_argument(
        "--pinv", type=Path,
        help="Path to the pseudoinverse trajectory (.npy)",
    )
    parser.add_argument(
        "--qp", type=Path,
        help="Path to the QP trajectory (.npy)",
    )
    parser.add_argument(
        "--output",
        default="build/ik_joint_diagnostics.png",
    )
    parser.add_argument(
        "--show", action="store_true",
    )

    args = parser.parse_args()

    trajectories = {}

    if args.pinv is not None:
        trajectories["Pseudoinverse"] = np.load(args.pinv)

    if args.qp is not None:
        trajectories["QP"] = np.load(args.qp)

    if not trajectories:
        parser.error("Provide --pinv and/or --qp trajectory files")

    plot_joint_diagnostics(
        trajectories,
        filename=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
