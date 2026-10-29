"""Robot: the real-robot backend, Microban. See README.md.

Where Microban and its robot model are (microban.py), the climbing model's balls, the limbs'
contact points, read from the model (hooks.py: `from rcr.robot import hooks`), and Microban's
observation and targets to and from our types (observation.py), for the sim and the robot.
Pi-safe.
"""

from rcr.robot.observation import to_state, to_targets

__all__ = ["to_state", "to_targets"]
