#!/bin/sh
# Put our Pi code on a Raspberry Pi, from the laptop, and optionally start it there [Ours].
#
#   sh scripts/to_pi.sh pi@172.20.10.8          # build the bundle, copy it to ~/rcr on the Pi
#   sh scripts/to_pi.sh pi@172.20.10.8 --run    # ... and (re)start the receiver there (Ctrl-C stops it)
#
# The user before @ is the Pi's: pi on our test Pi, user on Microban's image. It asks for the
# Pi's password (once to copy, once more with --run). Works from any folder. Then, on the laptop:
#   uv run scripts/link_pi.py check --pi 172.20.10.8
set -e

if [ -z "$1" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
HOST="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"  # the repository, wherever this is run from

echo "1. building the bundle"
(cd "$ROOT" && uv run scripts/link_pi.py bundle) || { echo "bundle failed"; exit 1; }

echo "2. copying it to $HOST:~/rcr"
rsync -av "$ROOT/dist/raspberry_pi/" "$HOST:rcr/" || {
    echo "copy failed: is the Pi on, on the same network, and is \"$HOST\" right (user@address)?"
    exit 1
}
echo "copied. On the Pi: sh ~/rcr/run.sh"

if [ "$2" = "--run" ]; then
    echo "3. starting the receiver on $HOST, replacing one already running (Ctrl-C to stop)"
    # [.] so that the pattern does not match this command line itself, only the receiver
    ssh -t "$HOST" 'pkill -f "rcr[.]link[.]raspberry_pi" && sleep 1; sh ~/rcr/run.sh'
fi
