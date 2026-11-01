#!/usr/bin/env bash
# Run the optimiser (the TO) with every setting in one place: change the values below, then
#     bash src/rcr/optimiser/run.sh          solve, play each move in the organisers' sim, save
#     bash src/rcr/optimiser/run.sh --play   play the saved trajectory again, without solving
# The trajectory is saved next to the plan, as trajectory_<the plan's name>.json, after every
# solved move: build/plan_N3.json gives build/trajectory_plan_N3.json. A plan in data/results/ is
# tracked by git, so a run elsewhere (the cluster) can commit its trajectory there to pull here.
# Anything else after run.sh goes to the runner too (bash src/rcr/optimiser/run.sh --help lists
# its options).
# The full problem (constraints and cost) is in README.md, "One move: the problem".
set -euo pipefail
cd "$(dirname "$0")/../../.."  # the repository root

# --- What to solve ---------------------------------------------------------------------------------
# PLAN:  the contact plan (scripts/plan_sim_wall.py --save): the stances, so where every move
#        starts and ends. Required.
# GUESS: optional, a motion for the same plan (.npz, q: T x 26 at 50 Hz, e.g. an IK trajectory):
#        the solver's starting guess for each move it has too (from the same stance to the same
#        stance), so it converges faster; the other moves start as usual. "" for none. With
#        W_TRACK > 0 the result is also kept near it (then it must have every move).
# START: optional, a trajectory PLAN continues (.json, as saved here): PLAN starts at its last
#        pose (it must be on PLAN's first stance), and the saved trajectory is it followed by
#        PLAN's moves. "" to start from a static pose of PLAN's first stance.
# N2, moves 34-55 (plan_N2_rest.json, extended to the top of the wall in small steps, see the
# optimiser README): no IK there, so no GUESS (W_TRACK 0), continuing the first 34 moves solved
# with it (plan_N2.json and wall-2-ik.npz, saved as trajectory_plan_N2_local.json).
PLAN="data/results/to_n2/plan_N2_rest.json"
GUESS=""
START="data/results/to_n2/trajectory_plan_N2_local.json"

# --- Constraints -----------------------------------------------------------------------------------
TAU_LIMIT=0.9   # motor torque limit, as a fraction of the servo's 0.605 N·m
N_SAMPLES=12     # samples per move (more: finer, slower)

# --- Cost weights (0 turns a term off) -----------------------------------------------------------
W_TRACK=50        # stay near GUESS's motion (0: GUESS is only the starting guess)
W_TAU=1          # low motor torques
W_SMOOTH=1       # smooth motion between samples
W_NOMINAL=0.3      # posture: legs near a hand-designed pose (0: trust the reference instead)
W_SPINE=2        # posture: spine leaning 25 deg from vertical
W_SWING=5000     # moving ball along the swing path (up, out, across, in, down)
W_KEEP_OUT=20000  # body out of the cups' layer
W_HOOK=100         # hooks pointing up from their balls (into the cups from above)

# --- Output ----------------------------------------------------------------------------------------
HEADLESS=0       # 1: no window

# ---------------------------------------------------------------------------------------------------
name="$(basename "$PLAN")"
out="$(dirname "$PLAN")/trajectory_${name%.*}.json"  # plan_N3.json -> trajectory_plan_N3.json
mode=--save
if [[ "${1:-}" == --play ]]; then
    shift
    mode=--replay
fi

args=("$PLAN")
if [[ -n "$START" && "$mode" == --save ]]; then
    args+=(--start "$START")
fi
if [[ -n "$GUESS" ]]; then
    args+=(--guess "$GUESS")
else
    W_TRACK=0  # nothing to follow
fi
args+=(--tau-limit "$TAU_LIMIT" --samples "$N_SAMPLES")
for setting in w_track="$W_TRACK" w_tau="$W_TAU" w_smooth="$W_SMOOTH" w_nominal="$W_NOMINAL" \
    w_spine="$W_SPINE" w_swing="$W_SWING" w_keep_out="$W_KEEP_OUT" w_hook="$W_HOOK"; do
    args+=(--set "$setting")
done
args+=("$mode" --out "$out")
if [[ "$HEADLESS" == 1 ]]; then args+=(--headless); fi
python=python
if [[ "$(uname)" == Darwin ]]; then python=mjpython; fi  # macOS: mjpython for the MuJoCo window

echo "uv run $python tests/test_optimiser.py ${args[*]} $*"
uv run "$python" tests/test_optimiser.py "${args[@]}" "$@"
