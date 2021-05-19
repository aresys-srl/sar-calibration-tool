# -*- coding: utf-8 -*-

"""
Earth Explorer orbit module
"""

import logging

log = logging.getLogger(__name__)

import numpy as np
import os
import coda

from arepytools.timing.precisedatetime import PreciseDateTime

from sct.orbit.orbit import Orbit, EOrbitType


class EEOrbit(Orbit):
    """EEOrbit class"""

    orbit_type_dict = {
        "AUX_PREORB": EOrbitType.predicted.value,
        "AUX_RESORB": EOrbitType.restituted.value,
        "AUX_POEORB": EOrbitType.precise.value,
    }

    def __init__(self, orbit_file):
        """Initialise EEOrbit object

        :param orbit_file: Path to orbit file
        :type orbit_file: str
        """

        log.debug("Initialize orbit file")

        Orbit.__init__(self, orbit_file)

        self.__read_orbit()

    def __read_orbit(
        self,
    ):
        """Read orbit file

        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.debug("Read orbit file")

        # Set orbit file name
        self.name = os.path.basename(self.orbit_file)

        # Read orbit file
        # - Open orbit file
        file_handle = coda.open(self.orbit_file)

        # - Fetch content
        file_data = coda.fetch(file_handle)

        # - Set orbit metadata
        fixed_header = file_data.Earth_Explorer_File.Earth_Explorer_Header.Fixed_Header
        self.mission = fixed_header.Mission
        self.type = self.orbit_type_dict.get(fixed_header.File_Type, EOrbitType.unknown.value)
        self.start_time = PreciseDateTime().set_from_utc_string(
            fixed_header.Validity_Period.Validity_Start[4:] + ".000000"
        )
        self.stop_time = PreciseDateTime().set_from_utc_string(
            fixed_header.Validity_Period.Validity_Stop[4:] + ".000000"
        )

        # - Set orbit data
        list_of_osvs = file_data.Earth_Explorer_File.Data_Block.List_of_OSVs
        sv_count = len(list_of_osvs.OSV)
        self.position_sv = np.zeros((sv_count, 3))
        self.velocity_sv = np.zeros((sv_count, 3))
        for sv in range(sv_count):
            self.position_sv[sv][0] = float(list_of_osvs.OSV[sv].X)
            self.position_sv[sv][1] = float(list_of_osvs.OSV[sv].Y)
            self.position_sv[sv][2] = float(list_of_osvs.OSV[sv].Z)
            self.velocity_sv[sv][0] = float(list_of_osvs.OSV[sv].VX)
            self.velocity_sv[sv][1] = float(list_of_osvs.OSV[sv].VY)
            self.velocity_sv[sv][2] = float(list_of_osvs.OSV[sv].VZ)
        self.reference_time = PreciseDateTime().set_from_utc_string(list_of_osvs.OSV[0].UTC[4:])
        self.delta_time = PreciseDateTime().set_from_utc_string(list_of_osvs.OSV[1].UTC[4:]) - self.reference_time

        # - Close orbit file
        status = coda.close(file_handle)

        return True
