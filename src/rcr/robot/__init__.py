"""Robot: the real-robot backend, Microban. See README.md.

Where Microban and its robot model are (microban.py), and the climbing model's balls, the limbs'
contact points (hooks.py), shared by the optimiser and the sim. Pi-safe.
"""

from rcr.robot.hooks import BALL_RADIUS, HOOKS, Hook, ball_centre

__all__ = ["BALL_RADIUS", "HOOKS", "Hook", "ball_centre"]
