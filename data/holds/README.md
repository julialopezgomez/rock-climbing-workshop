# holds

The hold itself is the organisers' simulator's: `climbing_sim/assets/hold/hold_visual.stl` in `third_party/simulator`
(in m, origin at the socket centre, the wall frame's axes; spherical socket of 12.7 mm radius). The optimiser reads it
from there (`rcr.optimiser.collision.HOLD_STL`); nothing of it is copied here.

`hold_pieces_<sha>_<threshold>.npz`: that mesh cut into convex pieces (CoACD), which the optimiser's collision check
uses inside the hold [Ours]. It is kept here, not rebuilt on each machine, because CoACD gives slightly different
pieces on another machine, and with them slightly different solutions. If the simulator's mesh changes, the optimiser
makes a new one here on its first run (~20 s): commit it.
