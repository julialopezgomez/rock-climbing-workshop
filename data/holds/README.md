# holds

`hold.stl`: the provided hold [Rules v1.1 §4, Figure 3], copied unchanged from the organisers' rules repo,
[`figures/hold.stl`](https://github.com/rock-climbing-robots/rules/blob/eea6f34117414537b4c1c54143b1afb810fdf889/figures/hold.stl)
at commit `eea6f34` (2026-10-20). They mesh it from their CAD (`assets/hold.step`, not published) with gmsh, ~20k triangles,
and draw Figure 3 from it. sha256 `8c8df8c61e96921d95b4f69e85f6c9bb35e0cb6cd6386308a54277f5ebc5776e`.

Frame (their `figures/gen_hold.py`), in **mm**: origin at the socket centre, +y towards the wall, back face at y = +26,
bolt holes at x = ±15, z = +14. The spherical socket has a 12.7 mm radius (measured on the mesh).

If the organisers update it, copy the new file over this one and update the commit above. The official simulator
(rock-climbing-robots/simulator) should ship the same mesh.
