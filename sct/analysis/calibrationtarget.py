# -*- coding: utf-8 -*-

"""
Calibration target module
"""

import numpy as np
from shapely.geometry import Point, Polygon

from arepytools.geometry.conversions import xyz2llh, llh2xyz


class CalibrationTarget:
    """CalibrationTarget class"""

    def __init__(
        self, id, type, latitude, longitude, altitude, rcs, delay, plate, survey_time, validity_start, validity_stop
    ):
        """Initialise CalibrationTarget object

        :param id: Calibration target identifier
        :type id: str
        :param type: Calibration target type
        :type type: str
        :param latitude: Calibration target latitude [deg]
        :type latitude: float
        :param longitude: Calibration target longitude [deg]
        :type longitude: float
        :param altitude: Calibration target altitude [m]
        :type altitude: float
        :param rcs: Calibration target RCS as dictionary of 4 float values, one per polarisation [dB]
        :type rcs: dict
        :param delay: Calibration target delay [s]
        :type delay: float
        :param plate: Calibration target plate
        :type plate: str
        :param survey_time: Calibration target survey time [UTC]
        :type survey_time: PreciseDateTime
        :param validity_start: Calibration target validity start time [UTC]
        :type validity_start: PreciseDateTime
        :param validity_stop: Calibration target validity stop time [UTC]
        :type validity_stop: PreciseDateTime
        """

        self.id = id
        self.type = type
        self.latitude = latitude  # [deg]
        self.longitude = longitude  # [deg]
        self.altitude = altitude  # [deg]
        self.rcs = rcs  # [dB]
        self.delay = delay  # [s]
        self.plate = plate
        self.survey_time = survey_time  # [Utc]
        self.validity_start = validity_start  # [Utc]
        self.validity_stop = validity_stop  # [Utc]

        self.xyz = self.__compute_geographical_coordinates()  # [m,m,m]

    def __str__(self):
        """CalibrationTarget class __str__ function

        :return: Calibration target description and main parameters
        :rtype: str
        """

        s = (
            "Calibration Target: {} ({})\n".format(self.id, self.type)
            + "Lat\\Lon\\Alt: {}deg - {}deg - {}m\n".format(self.latitude, self.longitude, self.altitude)
            + "Validity: from {} to {}".format(self.validity_start, self.validity_stop)
        )

        return s

    def __compute_geographical_coordinates(
        self,
    ):
        """Compute calibration target XYZ coordinates starting from LLH coordinates

        :return: Calibration target XYZ coordinates as numpy array of size 3x1
        :rtype: numpy.ndarray
        """

        xyz = llh2xyz([[self.latitude / 180 * np.pi], [self.longitude / 180 * np.pi], [self.altitude]])

        return xyz

    def update_coordinates(self, xyz_updated):
        """Update calibration target coordinates

        :param xyz_updated: XYZ coordinates as numpy array of size 3x1
        :type xyz_updated: numpy.ndarray
        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        self.xyz = xyz_updated.reshape(3, 1)

        llh_updated = xyz2llh(xyz_updated)
        self.latitude = float(llh_updated[0] / np.pi * 180)
        self.longitude = float(llh_updated[1] / np.pi * 180)
        self.altitude = float(llh_updated[2])

        return True

    def is_visible(self, sar_product_footprint):
        """Check if calibration target is seen by SAR product, i.e. if it overlaps with its footprint

        :param sar_product_footprint: SAR product footprint as numpy array of size Nx2
        :type sar_product_footprint: numpy.ndarray
        :return: Status (True if yes, False otherwise)
        :rtype: bool
        """

        point = Point(self.latitude, self.longitude)
        polygon = Polygon(sar_product_footprint)

        return point.within(polygon)

    def is_valid(self, sar_product_start_time):
        """Check if calibration target is valid when seen by SAR product

        :param sar_product_start_time: SAR product start time [UTC]
        :type sar_product_start_time: PreciseDateTime
        :return: Status (True if yes, False otherwise)
        :rtype: bool
        """

        return (sar_product_start_time >= self.validity_start) and (sar_product_start_time <= self.validity_stop)
