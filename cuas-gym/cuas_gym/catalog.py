"""Threat drones of the CUAS-Gym instances, as described to the language model and used by the geometric heuristic.

A frozen copy of the three non-cooperative entries of the drone library of cuas_gym.physics (the only ones the instances use), so
that scoring and prompting do not need the physics module. ridPt / c2Pt are None when the drone has no Remote ID broadcast /
no radio emissions.
"""
DRONES = [
    {"key": "fpv", "name": "FPV / racing quad (no Remote ID)", "rcs": 0.01, "dc": 0.25, "ridPt": None, "c2Pt": 25},
    {"key": "fiber", "name": "Fiber-optic FPV (no radio at all)", "rcs": 0.01, "dc": 0.25, "ridPt": None, "c2Pt": None},
    {"key": "owa", "name": "Large fixed-wing, combustion engine (no Remote ID)", "rcs": 0.5, "dc": 1.5, "ridPt": None, "c2Pt": 30},
]
