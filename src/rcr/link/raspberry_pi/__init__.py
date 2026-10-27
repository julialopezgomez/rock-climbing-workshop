"""The Raspberry Pi's side of the link: RobotLink, and a receiver to run there. See README.md.

    PYTHONPATH=src python3 -m rcr.link.raspberry_pi     # on the Pi: receive, save, report

Pi-safe: stdlib only, plus rcr.types and rcr.io.
"""

from rcr.link.raspberry_pi.robot_link import RobotLink

__all__ = ["RobotLink"]
