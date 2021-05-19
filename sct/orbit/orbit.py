# -*- coding: utf-8 -*-

"""
Orbit module
"""

import enum

from arepytools.io.metadata import EOrbitDirection


class EOrbitType(enum.Enum):
    """EOrbitType class"""

    downlink = "DOWNLINK"
    predicted = "PREDICTED"
    restituted = "RESTITUTED"
    precise = "PRECISE"
    unknown = "UNKNOWN"


class Orbit:
    """Orbit class"""

    def __init__(self, orbit_file):
        """Initialise Orbit object

        :param orbit_file: Path to orbit file
        :type orbit_file: str
        """

        self.orbit_file = orbit_file

        self.name = None
        self.mission = None
        self.type = None
        self.start_time = None
        self.stop_time = None

        self.position_sv = []
        self.velocity_sv = []
        self.reference_time = None
        self.delta_time = None

    def get_orbit_direction(self, current_time):
        """Get orbit direction

        :param current_time: Current time [UTC]
        :type current_time: PreciseDateTime
        :return: Orbit direction
        :rtype: str
        """

        if self.name is None:
            raise RuntimeError("Orbit not initialized")

        current_ind = int((current_time - self.reference_time) / self.delta_time)
        if self.velocity_sv[current_ind][2] > 0:
            return EOrbitDirection.ascending.value
        else:
            return EOrbitDirection.descending.value
