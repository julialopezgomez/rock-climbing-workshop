"""Plain local report for physical RGB replay; no benchmark accuracy is inferred."""

import html
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .real_recordings import NAMES
from .src.calibration import project_wall


def build_report(root, images, cameras, maps, fused, summary):
    root = Path(root)
    masking = "With robot masking, the " if summary["sam3"].get("completed") else "The "
    learned = summary["learned_frontend"]
    if not learned.get("completed"):
        learned_result = "The learned detector was not run."
    elif learned.get("multiview_frontend_supported"):
        learned_result = (
            "The learned detector supplies observations in both views; "
            "pose verification remains separate."
        )
    else:
        learned_result = (
            "The learned detector did not supply reliable observations in both cameras."
        )
    for state, result in maps.items():
        for eye, side in enumerate(["l", "r"]):
            name = side + "_" + state
            image = images[name].copy()
            for hold in result["holds"]:
                if eye not in hold["observed_by"]:
                    continue
                uv = project_wall(
                    [hold["socket_hypothesis_wall_m"]],
                    cameras[name]["K"],
                    cameras[name]["T_camera_wall"],
                )[0]
                color = (20, 255, 90) if hold["image_supported"] else (255, 180, 30)
                centre = tuple(np.round(uv).astype(int))
                cv2.circle(image, centre, 7, color, 1)
            Image.fromarray(image).save(root / f"{name}_holds.jpg", quality=95)
    camera_rows = "".join(
        f"<tr><td>{name}</td><td>{cameras[name]['camera_position_wall_m'][0]:.3f}</td>"
        f"<td>{cameras[name]['camera_position_wall_m'][1]:.3f}</td>"
        f"<td>{cameras[name]['camera_position_wall_m'][2]:.3f}</td>"
        f"<td>{cameras[name]['reprojection_rms_px']:.3f}</td>"
        f"<td>{cameras[name]['inliers']}</td></tr>"
        for name in NAMES
    )
    map_rows = "".join(
        f"<tr><td>{state}</td><td>{result['view_counts'][0]}</td>"
        f"<td>{result['view_counts'][1]}</td><td>{len(result['holds'])}</td>"
        f"<td>{result['image_supported']}</td></tr>"
        for state, result in maps.items()
    )
    figures = []
    for state in ["without_rob", "with_rob"]:
        figures.append(
            f"<h3>{html.escape(state)}</h3><div class='pair'>"
            + "".join(
                f"<figure><img src='{side}_{state}_holds.jpg'>"
                f"<figcaption>{side}: green = CAD/image support; "
                "amber = rejected hypothesis.</figcaption></figure>"
                for side in ["l", "r"]
            )
            + "</div>"
        )
    if summary["robot_trunk"].get("completed"):
        figures.append(
            "<h3>Rigid-trunk hypothesis</h3><div class='pair'>"
            + "".join(
                f"<figure><img src='{side}_with_rob_trunk_detail.jpg'>"
                "<figcaption>Rendered rigid CAD in green; "
                "no moving-link estimate.</figcaption></figure>"
                for side in ["l", "r"]
            )
            + "</div>"
        )
    root_result = summary["robot_trunk"]
    root_description = (
        "A root hypothesis was fitted to both views. Translation in the provisional wall frame: "
        + ", ".join(f"{v:.3f}" for v in root_result["translation_m"])
        + f" m. Image fitting loss: {root_result['image_loss']:.3f}; this is not a position error."
        if root_result.get("completed")
        else html.escape(root_result.get("error", "Unavailable"))
    )
    page = f"""<!doctype html><html lang='en'><meta charset='utf-8'>
<title>Physical RGB perception replay</title>
<style>
body{{font:18px/1.5 serif;max-width:1150px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse;width:100%;margin:1em 0}}
th,td{{border-bottom:1px solid #aaa;padding:.35em;text-align:left}}
.pair{{display:flex;gap:1em}}figure{{margin:0;flex:1;min-width:0}}img{{width:100%}}
figcaption{{font-size:.85em}}code{{font: .9em monospace}}
@media(max-width:700px){{.pair{{display:block}}}}</style>
<h1>Physical RGB perception replay</h1>
<p>Four independently recorded color streams, processed with the board's hole lattice,
calibrated RGB intrinsics, cup CAD and an experimental rigid-trunk fit.</p>
<p><strong>These are image-supported hypotheses, not measured accuracy results.</strong>
No surveyed hold map, robot pose or joint log was supplied.
The board/grid origin remains unverified;
<code>valid_for_control=false</code> and <code>team_pose=null</code>.</p>
<h2>Camera positioning</h2>
<p>Color intrinsics came from the bag's RGB stream. The separate calibration exports describe
internal infrared cameras and were not used as external stereo calibration. Each recording
received its own pose because the right camera moved.</p>
<table><tr><th>Recording</th><th>x (m)</th><th>y (m)</th><th>z (m)</th>
<th>RMS (px)</th><th>Grid inliers</th></tr>{camera_rows}</table>
<p>Axes: x right, y up, z out from the wall. Dimensions and hole pitch are explicit priors:
{summary["geometry"]["wall_width_m"]:.3f} × {summary["geometry"]["wall_height_m"]:.3f} m,
{summary["geometry"]["hole_pitch_m"] * 1000:.1f} mm pitch. The first-hole offsets must be measured.
The residual measures grid reprojection, not absolute camera-position error.</p>
<h2>Cup mapping</h2><table><tr><th>Pair</th><th>Left proposals</th>
<th>Right proposals</th><th>Fitted hypotheses</th>
<th>Image-supported</th></tr>{map_rows}</table>
<p>Touching color components are split before upright CAD fitting. Source cup scale was checked
against alternatives. SAM 3 robot masks, when available, exclude robot parts from cup proposals.
The static fused map retains {len(fused["holds"])} image-supported hypotheses;
{fused["reobserved"]} empty-wall hypotheses were associated with robot-present observations.
Unseen holds are not filled in from a route. Socket and mounting offsets remain CAD assumptions.</p>
{"".join(figures)}
<h2>Robot pose</h2><p>{root_description}</p>
<p>{learned_result} The optional rigid-trunk
fit uses only root-body geometry and RGB support; it does not read or estimate joints.
The rope crosses the printed label, so normal fiducial decoding failed. Border geometry provides
an initializer with uncertain orientation.
A measured marker-to-trunk transform was not supplied.</p>
<h2>Time and scope</h2><p>Selected frames differ by
{summary["capture_separation_s"]["without_rob"]:.3f} s
without the robot and {summary["capture_separation_s"]["with_rob"]:.3f} s with it. Static wall/robot
assumptions permit an offline experiment; they do not establish synchronized exposure or a
valid live state. Within-recording camera repeatability is
saved separately from metric accuracy.</p>
<p><a href='summary.json'>Full results</a> · <a href='wall_map.json'>Static cup map</a> ·
<a href='label_hypothesis.json'>Label geometry</a> · <a href='trunk.json'>Trunk hypothesis</a> ·
<a href='real_images_report.pdf'>PDF</a></p></html>"""
    (root / "overview.html").write_text(page)
    tex_camera = "\n".join(
        f"{name.replace('_', r'\_')} & "
        f"{cameras[name]['reprojection_rms_px']:.3f} & {cameras[name]['inliers']} \\\\"
        for name in NAMES
    )
    tex = (
        r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage{lmodern,graphicx,booktabs,microtype}
\usepackage[margin=25mm]{geometry}\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Physical RGB Perception Replay},pdfauthor={Ege Gursoy}}
\title{Physical RGB Perception Replay}\author{Ege Gursoy}\date{30 October 2026}
\begin{document}\maketitle
Four RGB recordings were processed: one view with the robot and one empty-wall view
from each camera. Color intrinsics were read from the recordings. Camera poses were
estimated separately from the perforated board, using declared dimensions and hole pitch.
The infrared calibration exports do not describe the external camera pair.
\section{Camera and cup results}
\begin{center}\begin{tabular}{lrr}\toprule
Recording & Grid RMS (pixels) & Inlier holes\\\midrule
"""
        + tex_camera
        + r"""\bottomrule\end{tabular}\end{center}
The board is partly cropped and has no detected corner markers. A local hole grid is
indexed and expanded across the image, followed by planar pose fitting. The metric
origin remains conditional on declared board dimensions and provisional first-hole offsets.
A low image residual does not establish absolute camera accuracy.

Touching cup components are separated before multiview CAD fitting. The empty-wall
pair gives """
        + str(maps["without_rob"]["image_supported"])
        + r""" image-supported cup hypotheses.
"""
        + masking
        + "robot-present pair gives "
        + str(maps["with_rob"]["image_supported"])
        + r""".
These are support counts, not precision or recall. No reference cup map was supplied.
\begin{figure}[htbp]\centering
\includegraphics[width=.49\linewidth]{l_without_rob_holds.jpg}\hfill
\includegraphics[width=.49\linewidth]{r_without_rob_holds.jpg}
\caption{Empty-wall cup overlays. Green circles pass the image/CAD support gate;
amber circles mark rejected hypotheses. Both camera poses are estimated from RGB.}
\end{figure}
\clearpage
\section{Robot hypothesis and limitations}
"""
        + learned_result
        + r""" The supplied images show rope occlusion of the printed label.
Visible label
geometry initializes a separate rigid-trunk CAD fit. This fit excludes moving links,
reads no encoder angles and estimates no joints or endpoints.
"""
    )
    if root_result.get("completed"):
        tex += r"""\begin{figure}[htbp]\centering
\includegraphics[width=.35\linewidth]{l_with_rob_trunk_detail.jpg}\hfill
\includegraphics[width=.35\linewidth]{r_with_rob_trunk_detail.jpg}
\caption{Experimental trunk geometry rendered in green at the fitted pose. No
surveyed root pose is available; visual agreement does not measure metric accuracy.}
\end{figure}
"""
    tex += (
        "The selected robot frames were captured "
        + f"{summary['capture_separation_s']['with_rob']:.3f}"
        + r""" seconds apart. Their use as a stereo observation assumes a stationary robot.
Neither synchronized exposure nor physical localization accuracy has been verified.
The right camera also moved between empty-wall and robot recordings.
The output therefore retains \texttt{valid\_for\_control=false} and publishes no live
team pose. The next measurements are board/grid offsets, a synchronized RGB pair,
encoder data if articulated fitting is required, and an independent pose reference.
\end{document}
"""
    )
    (root / "real_images_report.tex").write_text(tex)
    if shutil.which("pdflatex"):
        with (root / "report_build.log").open("w") as log:
            for _ in range(2):
                result = subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "real_images_report.tex",
                    ],
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                if result.returncode:
                    raise RuntimeError("Physical report compilation failed; see report_build.log")
