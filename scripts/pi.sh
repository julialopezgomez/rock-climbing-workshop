#!/bin/sh
# Everything with the Raspberry Pi, from the laptop [Ours]. TARGET is user@address, e.g.
# pi@172.20.10.8 (pi on our test Pi, user on Microban's image). Works from any folder, and
# TARGET may come before or after the option.
#
#   sh scripts/pi.sh --install TARGET          copy our code to ~/rcr on the Pi (after code changes)
#   sh scripts/pi.sh --install TARGET --run    ... and (re)start the receiver there, in this window
#   sh scripts/pi.sh --check   TARGET          test the link (round trip, upload, state back)
#   sh scripts/pi.sh --send    TARGET [FILE]   send a whole trajectory once (default: the example)
#   sh scripts/pi.sh --stream  TARGET [FILE]   send it, start, then each move saved into it (Ctrl-C)
#   sh scripts/pi.sh --start   TARGET          start / stop the climb
#   sh scripts/pi.sh --stop    TARGET
#
# --check, --send, --stream, --start and --stop need the receiver running on the Pi: there,
# sh ~/rcr/run.sh (or --install TARGET --run here). --stream follows the file: run the optimiser
# with --save (tests/test_optimiser.py --hard --save) and each move it saves goes to the robot.
# More in src/rcr/link/README.md.
set -e

usage() {
    awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"  # the comment above
    exit 1
}
[ $# -ge 2 ] || usage
case "$1" in
--*) ACTION="$1" TARGET="$2" ;;
*) TARGET="$1" ACTION="$2" ;;  # TARGET first: pi@172.20.10.8 --install
esac
shift 2
ADDRESS="${TARGET#*@}"  # what the link connects to: the address without the user
ROOT="$(cd "$(dirname "$0")/.." && pwd)"  # the repository, wherever this is run from

link() {  # scripts/link_pi.py in the repository's environment; FILE paths stay as given
    uv run --project "$ROOT" "$ROOT/scripts/link_pi.py" "$@"
}

case "$ACTION" in
--install)
    echo "building the bundle"
    link bundle
    echo "copying it to $TARGET:~/rcr (asks for the Pi's password)"
    rsync -av "$ROOT/dist/raspberry_pi/" "$TARGET:rcr/" || {
        echo "copy failed: is the Pi on, on the same network, and is \"$TARGET\" right (user@address)?"
        exit 1
    }
    if [ "$1" = "--run" ]; then
        echo "starting the receiver on $TARGET, replacing one already running (Ctrl-C to stop)"
        # [.] so that the pattern does not match this command line itself, only the receiver
        ssh -t "$TARGET" 'pkill -f "rcr[.]link[.]raspberry_pi" && sleep 1; sh ~/rcr/run.sh'
    else
        echo "installed. A receiver already running keeps the old code: restart it on the Pi"
        echo "(Ctrl-C, then sh ~/rcr/run.sh), or here: sh scripts/pi.sh --install $TARGET --run"
    fi
    ;;
--check) link check --pi "$ADDRESS" "$@" ;;
--send) link send --pi "$ADDRESS" "$@" ;;
--stream) link stream --pi "$ADDRESS" "$@" ;;
--start) link start --pi "$ADDRESS" ;;
--stop) link stop --pi "$ADDRESS" ;;
*) usage ;;
esac
