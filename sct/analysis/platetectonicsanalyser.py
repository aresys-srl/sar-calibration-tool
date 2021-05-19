# -*- coding: utf-8 -*-

"""
Plate tectonics analysis module
"""

import logging

log = logging.getLogger(__name__)

import numpy as np


class PlateTectonicsAnalyser:
    """PlateTectonicsAnalyser class"""

    # Absolute plate rotation poles (or angular velocities) defining ITRF2014-PMM
    # Source: Zuheir Altamimi et al., "ITRF2014 plate motion model", Geophysical Journal International, 2017
    itrf2014_plates_rotation_poles = {
        "ANTA": [-0.248, -0.324, 0.675],
        "ARAB": [1.154, -0.136, 1.444],
        "AUST": [1.510, 1.182, 1.215],
        "EURA": [-0.085, -0.531, 0.770],
        "INDI": [1.154, -0.005, 1.454],
        "NAZC": [-0.333, -1.544, 1.623],
        "NOAM": [0.024, -0.694, -0.063],
        "NUBI": [0.099, -0.614, 0.733],
        "PCFC": [-0.409, 1.047, -2.169],
        "SOAM": [-0.270, -0.301, -0.140],
        "SOMA": [-0.121, -0.794, 0.884],
    }  # [milliarcsec/yr]

    def __init__(self, xyz_coordinates, reference_plate, reference_time):
        """Initialise PlateTectonicsAnalyser object

        :param xyz_coordinates: Calibration target XYZ coordinates as numpy array of size 3x1
        :type xyz_coordinates: numpy.ndarray
        :param reference_plate: Calibration target reference plate
        :type reference_plate: str
        :param reference_time: Calibration target reference time [UTC]
        :type reference_time: PreciseDateTime
        """

        self.xyz_coordinates = xyz_coordinates
        self.reference_plate = reference_plate
        self.reference_time = reference_time

    def __get_plate_displacement_parameters(
        self,
    ):
        """Get plate displacement parameters

        :return: Plate rotation angular velocities [rad/s]
        :rtype: float
        """

        # Get plate displacement values from reference LUT
        # TODO Use shape file (or similar) accessed through self.xyz_coordinates instead of dictionary
        if self.reference_plate not in self.itrf2014_plates_rotation_poles:
            raise ValueError("Not recognized plate value: {}".format(self.reference_plate))

        omega = np.array(self.itrf2014_plates_rotation_poles[self.reference_plate])

        # Convert unit of measure [milliarcsec/yr -> rad/s]
        omega = omega * (1 / 1000 * 1 / 3600 * np.pi / 180) / (3600 * 24 * 365.25)

        return omega

    def get_updated_coordinates(self, acquisition_time):
        """Compute updated calibration target XYZ coordinates after the application of plate tectonics correction

        :param acquisition_time: Acquisition time [UTC]
        :type acquisition_time: PreciseDateTime
        :return: Calibration target updated XYZ coordinates as numpy array of size 3x1
        :rtype: numpy.ndarray
        """

        # Derive plate displacement parameters
        log.debug("Read plate displacement parameters")
        omega = self.__get_plate_displacement_parameters()

        # Compute delta time between acquisition and reference one
        delta_time = acquisition_time - self.reference_time

        # Loop over input coordinates
        log.debug("Compute updated target coordinates")
        n_points = self.xyz_coordinates.shape[1]
        xyz_coordinates_updated = np.zeros(self.xyz_coordinates.shape)
        for p in range(n_points):

            # Compute displacement velocity
            displacement_velocity = np.cross(omega, self.xyz_coordinates[:, p])

            # Compute updated coordinates
            xyz_coordinates_updated = self.xyz_coordinates[:, p] + displacement_velocity * delta_time

        return xyz_coordinates_updated
