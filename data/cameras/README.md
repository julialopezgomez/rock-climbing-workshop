# Real camera calibration

`realsense.json` stores the **color** intrinsics reported by
`sudo /opt/homebrew/bin/rs-enumerate-devices -c` for two D435i cameras at
640x480. `realsense-1280x720.json` stores their color intrinsics at 1280x720.
Each entry is labeled by the device serial number.

Run from the repository root with the OpenCV camera indices in **the same
order** as the JSON entries:

```sh
third_party/simulator/.venv/bin/python scripts/capture_robot_pose.py --cameras INDEX_FOR_243222071545 INDEX_FOR_233722072758 --save-dir /tmp/robot-pose-frames
```

The script reads this file by default. To change a camera, update its serial,
`camera_matrix`, and `distortion_coeffs` from the new device's **Color**
intrinsics at the resolution specified by `image_size`. Use
`--calibration data/cameras/realsense-1280x720.json` for 1280x720 capture.

OpenCV camera indices are not stable and this script cannot verify that an index
matches the configured serial. Check the saved frames whenever cameras are
connected or reordered; do not trust a pose from the wrong camera calibration.
Replace the `INDEX_FOR_...` placeholders with the current integer OpenCV indices.
The wall-marker positions and robot-marker mounting offset in
`scripts/robot_pose_aruco.py` also need to be measured for the real setup.
