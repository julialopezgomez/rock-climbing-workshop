# Code Template to potentially run 

```
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path("scripts").resolve()))
from overlay_wall_grid import calibration_for
from robot_pose_aruco import estimate_robot_pose

def locate(frames_bgr, serials):
    if not 1 <= len(frames_bgr) <= 2 or len(frames_bgr) != len(serials):
        raise ValueError("Provide one or two frames, with one serial per frame")

    images_rgb, matrices, distortions = [], [], []
    for frame, serial in zip(frames_bgr, serials, strict=True):
        if frame is None:
            raise ValueError("Could not read a frame")
        height, width = frame.shape[:2]
        K, distortion = calibration_for(serial, (width, height))
        images_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        matrices.append(K)
        distortions.append(distortion)

    return estimate_robot_pose(images_rgb, matrices, distortions)

# Repeat this part for every new pair:
pairs = [
    (["../test images/AruCo4_Color.png", "../test images/AruCo3_Color.png"],
     ["233722072758", "243222071545"]),
    # Add more pairs here. A one-image entry also works.
]

for paths, serials in pairs:
    frames = [cv2.imread(path) for path in paths]
    result = locate(frames, serials)
    print(result["method"], result["position_m"], result["quaternion_wxyz"])

```




