# -*- coding: utf-8 -*-

import numpy as np
from typing import Union

from arepytools import _utils
from arepytools.geometry.conversions import xyz2llh, llh2xyz
import arepytools.geometry.generalsarorbit as gso
from arepytools.io.metadata import StateVectors
from arepytools.math import axis as are_ax
from arepytools.timing.precisedatetime import PreciseDateTime

from sct.support.geometry_extended import look_to_slant_range


class GeneralSarOrbit_extended(gso.GeneralSarOrbit):
    def __init__(self, state_vectors):

        time_axis = are_ax.RegularAxis(
            (0, state_vectors.time_step, state_vectors.number_of_state_vectors), state_vectors.reference_time
        )
        super().__init__(time_axis, state_vectors.position_vector.reshape((state_vectors.position_vector.size,)))

    def get_velocity_ground(self, time_point, look_angle, reference_frame=None, time_span=1.0):

        look_angle = _utils.input_data_to_numpy_array_with_checks(look_angle, dtype=float)

        n = 11

        time_delta_axis = np.linspace(0, time_span, n)
        pp = np.zeros((3, look_angle.size, n))
        for i in range(n):
            pos = self.get_position(time_point + time_delta_axis[i])
            vel = self.get_velocity(time_point + time_delta_axis[i])
            _, _, _, p1, _ = look_to_slant_range(pos, vel, look_angle, reference_frame)
            pp[:, :, i] = p1

        velocity_ground = np.zeros(look_angle.shape)
        for a in range(look_angle.size):
            distance = 0
            for i in range(n - 1):
                p_diff = pp[:, a, i + 1] - pp[:, a, i]
                distance = distance + np.linalg.norm(p_diff)
            velocity_ground[a] = distance / time_span

        if velocity_ground.size == 1:
            velocity_ground = velocity_ground[0]

        return velocity_ground

    def get_incidence_angle(
        self,
        time_point,
        range_times,
        look_direction,
        geodetic_altitude=0.0,
        doppler_centroid=None,
        carrier_wavelength=None,
    ):

        range_times = _utils.input_data_to_numpy_array_with_checks(range_times, dtype=float)

        # Find ground point
        pg = self.sat2earth(
            time_point, range_times, look_direction, geodetic_altitude, doppler_centroid, carrier_wavelength
        )

        # Get sensor position
        psat = self.get_position(time_point).squeeze()

        # Get incidence angle
        incidence_angle = np.zeros((1, range_times.size))
        for nn in range(range_times.size):
            los = psat - pg[:, nn]
            los = los / np.linalg.norm(los)
            incidence_angle[0, nn] = np.arccos(np.dot(pg[:, nn] / np.linalg.norm(pg[:, nn]), los))

        # Convert radians to degrees
        incidence_angle = np.rad2deg(incidence_angle)

        if incidence_angle.size == 1:
            incidence_angle = incidence_angle[0, 0]

        return incidence_angle

    def get_sub_satellite_point(self, time_point, geodetic_altitude=0.0):

        time_point = _utils.input_data_to_numpy_array_with_checks(time_point)

        # Find sensor position
        pos_xyz = self.get_position(time_point)

        # Convert to geodetic coordinates
        pos_llh = xyz2llh(pos_xyz)

        # Go back to cartesian considering input altitude
        pos_llh[2, :] = np.zeros((1, time_point.size)) + geodetic_altitude
        sub_point = llh2xyz(pos_llh)

        return sub_point

    def get_nadir_pointing(self, time_point):

        time_point = _utils.input_data_to_numpy_array_with_checks(time_point)

        # Find satellite position
        pos_xyz = self.get_position(time_point)

        # Find satellite sub-point
        sub_point = self.get_sub_satellite_point(time_point)

        # Get nadir direction
        nadir = sub_point - pos_xyz

        # Normalize vector
        for nn in range(time_point.size):
            nadir[:, nn] = nadir[:, nn] / np.linalg.norm(nadir[:, nn])

        return nadir

    def get_look_angle(
        self,
        time_point,
        range_times,
        look_direction,
        geodetic_altitude=0.0,
        doppler_centroid=None,
        carrier_wavelength=None,
    ):

        range_times = _utils.input_data_to_numpy_array_with_checks(range_times, dtype=float)

        # Find ground point
        pg = self.sat2earth(
            time_point, range_times, look_direction, geodetic_altitude, doppler_centroid, carrier_wavelength
        )

        # Get sensor position
        psat = self.get_position(time_point).squeeze()

        # Get nadir direction
        nadir = self.get_nadir_pointing(time_point).squeeze()

        # Get look angle
        look_angle = np.zeros((1, range_times.size))
        for nn in range(range_times.size):
            los = pg[:, nn] - psat
            los = los / np.linalg.norm(los)
            look_angle[0, nn] = np.arccos(np.dot(nadir, los))

        # Convert radians to degrees
        look_angle = np.rad2deg(look_angle)

        if look_angle.size == 1:
            look_angle = look_angle[0, 0]

        return look_angle


def create_general_sar_orbit(state_vectors: StateVectors):

    return GeneralSarOrbit_extended(state_vectors)
