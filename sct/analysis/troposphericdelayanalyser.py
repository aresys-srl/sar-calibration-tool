# -*- coding: utf-8 -*-

"""
Tropospheric delay analysis module
"""

import logging

log = logging.getLogger(__name__)

import numpy as np
import os
from scipy.interpolate import interp1d, griddata

from arepytools.geometry.conversions import xyz2llh
from arepytools.timing.precisedatetime import PreciseDateTime

from sct.support.utils import compute_incidence_angle


class TroposphericDelayAnalyser:
    """TroposphericDelayAnalyser class"""

    def __init__(
        self,
        xyz_coordinates,
        troposphere_maps_dir,
        troposphere_maps_model="GRID",
        troposphere_maps_resolution="1x1",
        troposphere_maps_version="OP",
    ):
        """Initialise TroposphericDelayAnalyser object

        :param xyz_coordinates: Calibration targets XYZ coordinates as numpy array of size 3xN
        :type xyz_coordinates: numpy.ndarray
        :param troposphere_maps_dir: Path to troposphere maps folder
        :type troposphere_maps_dir: str
        :param troposphere_maps_model: Troposphere maps model, defaults to 'GRID'
        :type troposphere_maps_model: str, optional
        :param troposphere_maps_resolution: Troposphere maps resolution, defaults to '1x1'
        :type troposphere_maps_resolution: str, optional
        :param troposphere_maps_version: Troposphere maps version, defaults to 'OP'
        :type troposphere_maps_version: str, optional
        """

        self.xyz_coordinates = xyz_coordinates

        self.troposphere_maps_dir = troposphere_maps_dir
        self.troposphere_maps_model = troposphere_maps_model
        self.troposphere_maps_resolution = troposphere_maps_resolution
        self.troposphere_maps_version = troposphere_maps_version

        self.interpolation_method = "cubic"

    def __get_troposphere_map_file(self, acquisition_time):
        """Get troposphere map file starting from acquisition time. Returns error if file is not found

        :param acquisition_time: Acquisition time [UTC]
        :type acquisition_time: PreciseDateTime
        :return: Troposphere map filename
        :rtype: str
        """

        # Define troposphere map filename and path
        year = acquisition_time.year
        month = acquisition_time.month
        day = acquisition_time.day_of_the_month
        hour = acquisition_time.hour_of_day

        troposphere_maps_resolution_list = [self.troposphere_maps_resolution, "5x5"]
        troposphere_maps_version_list = [self.troposphere_maps_version, "EI"]
        for version in troposphere_maps_version_list:
            for resolution in troposphere_maps_resolution_list:
                troposphere_map_file = "VMF3_{0}{1:02}{2:02}.H{3:02}".format(year, month, day, hour)
                # --> path for download: https://vmf.geo.tuwien.ac.at/trop_products/<model>/<resolution>/VMF3/VMF3_<version>/year/<troposphere_map_file>
                troposphere_map_file = os.path.join(self.troposphere_maps_dir, troposphere_map_file)

                # Check if exists, otherwise try alternatives
                if os.path.isfile(troposphere_map_file):
                    return troposphere_map_file

        raise FileNotFoundError(
            "No troposphere map found for the {} date. File {} not found.".format(acquisition_time,
            os.path.basename(troposphere_map_file)),
        )

    def __read_troposphere_map_file(self, troposphere_map_file):
        """Read troposphere map file

        :param troposphere_map_file: Path to troposphere map file
        :type troposphere_map_file: str
        :return: Tuple: Troposphere map data and corresponding latitude and longitude axes, all as numpy arrays
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        # Read troposphere map file
        with open(troposphere_map_file, mode="rb") as file:
            # Read file removing header section
            lines = file.readlines()[7:]

            # Return lists of parameters
            n_lines = len(lines)
            tropo_map_lat_axis = np.zeros(n_lines)
            tropo_map_lon_axis = np.zeros(n_lines)
            a_h = np.zeros(n_lines)
            a_w = np.zeros(n_lines)
            zpd_h = np.zeros(n_lines)
            zpd_w = np.zeros(n_lines)
            for ind, line in enumerate(lines):
                tropo_map_lat_axis[ind], tropo_map_lon_axis[ind], a_h[ind], a_w[ind], zpd_h[ind], zpd_w[ind] = [
                    float(x) for x in line.split()
                ]

        # Shift latitude and longitude axes ([0,360]->[-180,180])
        tropo_map_lat_axis[tropo_map_lat_axis > 180] = tropo_map_lat_axis[tropo_map_lat_axis > 180] - 360
        tropo_map_lon_axis[tropo_map_lon_axis > 180] = tropo_map_lon_axis[tropo_map_lon_axis > 180] - 360

        return tropo_map_lon_axis, tropo_map_lat_axis, a_h, a_w, zpd_h, zpd_w

    def __read_spherical_harmonics_coefficients_file(self, spherical_harmonics_coefficients_file):
        """Read spherical harmonics coefficients file

        :param spherical_harmonics_coefficients_file: Path to spherical harmonics coefficients file
        :type spherical_harmonics_coefficients_file: str
        :return: Spherical harmonics coefficients as numpy array
        :rtype: numpy.ndarray
        """

        with open(spherical_harmonics_coefficients_file) as f:
            # Read file
            lines = f.readlines()

            # Return coefficients
            n_lines = len(lines)
            spherical_harmonics_coefficients = np.zeros((n_lines, 5))
            for ind, line in enumerate(lines):
                spherical_harmonics_coefficients[ind, :] = [float(x) for x in line.split()]

        return spherical_harmonics_coefficients

    def __read_grid_point_coordinates_file(self, grid_point_coordinates_file):
        """Read grid point coordinates file

        :param grid_point_coordinates_file: Path to grid point coordinates file
        :type grid_point_coordinates_file: str
        :return: Tuple: Grid points latitudes, longitudes and heights, all as numpy arrays
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        with open(grid_point_coordinates_file, mode="rb") as f:
            # Read file removing header section
            lines = f.readlines()
            lines = lines[14:]

            # Return coordinates
            n_lines = len(lines)
            grid_lat_axis = np.zeros(n_lines)
            grid_lon_axis = np.zeros(n_lines)
            grid_height = np.zeros(n_lines)
            for ind, line in enumerate(lines):
                _, grid_lat_axis[ind], grid_lon_axis[ind], grid_height[ind], _ = line.split()

        # Shift longitude axis ([0,360]->[-180,180])
        grid_lon_axis[grid_lon_axis > 180] = grid_lon_axis[grid_lon_axis > 180] - 360

        return grid_lat_axis, grid_lon_axis, grid_height

    def __interpolate_spherical_harmonics_coefficients(
        self, anm_bh, bnm_bh, anm_bw, bnm_bw, anm_ch, bnm_ch, anm_cw, bnm_cw, acquisition_time, lat, lon
    ):
        """Interpolate spherical harmonics coefficients

        :param anm_bh: Spherical harmonics coefficient anm_bh
        :type anm_bh: numpy.ndarray
        :param bnm_bh: Spherical harmonics coefficient bnm_bh
        :type bnm_bh: numpy.ndarray
        :param anm_bw: Spherical harmonics coefficient anm_bw
        :type anm_bw: numpy.ndarray
        :param bnm_bw: Spherical harmonics coefficient bnm_bw
        :type bnm_bw: numpy.ndarray
        :param anm_ch: Spherical harmonics coefficient anm_ch
        :type anm_ch: numpy.ndarray
        :param bnm_ch: Spherical harmonics coefficient bnm_ch
        :type bnm_ch: numpy.ndarray
        :param anm_cw: Spherical harmonics coefficient anm_cw
        :type anm_cw: numpy.ndarray
        :param bnm_cw: Spherical harmonics coefficient bnm_cw
        :type bnm_cw: numpy.ndarray
        :param acquisition_time: Acquisition time [UTC]
        :type acquisition_time: PreciseDateTime
        :param lat: Calibration target latitude [deg]
        :type lat: float
        :param lon: Calibration target longitude [deg]
        :type lon: float
        :return: Tuple: Interpolated spherical harmonics coefficients b_h, b_w, c_h, c_w, all as numpy arrays
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        # Reorganize input coefficients
        anm_bh_A0 = anm_bh[:, 0]
        anm_bh_A1 = anm_bh[:, 1]
        anm_bh_B1 = anm_bh[:, 2]
        anm_bh_A2 = anm_bh[:, 3]
        anm_bh_B2 = anm_bh[:, 4]
        bnm_bh_A0 = bnm_bh[:, 0]
        bnm_bh_A1 = bnm_bh[:, 1]
        bnm_bh_B1 = bnm_bh[:, 2]
        bnm_bh_A2 = bnm_bh[:, 3]
        bnm_bh_B2 = bnm_bh[:, 4]
        anm_bw_A0 = anm_bw[:, 0]
        anm_bw_A1 = anm_bw[:, 1]
        anm_bw_B1 = anm_bw[:, 2]
        anm_bw_A2 = anm_bw[:, 3]
        anm_bw_B2 = anm_bw[:, 4]
        bnm_bw_A0 = bnm_bw[:, 0]
        bnm_bw_A1 = bnm_bw[:, 1]
        bnm_bw_B1 = bnm_bw[:, 2]
        bnm_bw_A2 = bnm_bw[:, 3]
        bnm_bw_B2 = bnm_bw[:, 4]
        anm_ch_A0 = anm_ch[:, 0]
        anm_ch_A1 = anm_ch[:, 1]
        anm_ch_B1 = anm_ch[:, 2]
        anm_ch_A2 = anm_ch[:, 3]
        anm_ch_B2 = anm_ch[:, 4]
        bnm_ch_A0 = bnm_ch[:, 0]
        bnm_ch_A1 = bnm_ch[:, 1]
        bnm_ch_B1 = bnm_ch[:, 2]
        bnm_ch_A2 = bnm_ch[:, 3]
        bnm_ch_B2 = bnm_ch[:, 4]
        anm_cw_A0 = anm_cw[:, 0]
        anm_cw_A1 = anm_cw[:, 1]
        anm_cw_B1 = anm_cw[:, 2]
        anm_cw_A2 = anm_cw[:, 3]
        anm_cw_B2 = anm_cw[:, 4]
        bnm_cw_A0 = bnm_cw[:, 0]
        bnm_cw_A1 = bnm_cw[:, 1]
        bnm_cw_B1 = bnm_cw[:, 2]
        bnm_cw_A2 = bnm_cw[:, 3]
        bnm_cw_B2 = bnm_cw[:, 4]

        # Compute Legendre polynomials
        lon = lon * np.pi / 180  # Target unit vector
        lat = lat * np.pi / 180
        pdd = np.pi / 2 - lat
        x = np.sin(pdd) * np.cos(lon)
        y = np.sin(pdd) * np.sin(lon)
        z = np.cos(pdd)

        n_max = 12
        V = np.zeros((n_max + 2, n_max + 2))
        W = np.zeros((n_max + 2, n_max + 2))
        V[1, 1] = 1
        W[1, 1] = 0
        V[2, 1] = z * V[1, 1]
        W[2, 1] = 0

        for n in range(2, n_max + 1):
            V[n + 1, 1] = ((2 * n - 1) * z * V[n, 1] - (n - 1) * V[n - 1, 1]) / n
            W[n + 1, 1] = 0

        for m in range(1, n_max + 1):
            V[m + 1, m + 1] = (2 * m - 1) * (x * V[m, m] - y * W[m, m])
            W[m + 1, m + 1] = (2 * m - 1) * (x * W[m, m] + y * V[m, m])
            if m < n_max:
                V[m + 2, m + 1] = (2 * m + 1) * z * V[m + 1, m + 1]
                W[m + 2, m + 1] = (2 * m + 1) * z * W[m + 1, m + 1]
            for n in range(m + 2, n_max + 1):
                V[n + 1, m + 1] = ((2 * n - 1) * z * V[n, m + 1] - (n + m - 1) * V[n - 1, m + 1]) / (n - m)
                W[n + 1, m + 1] = ((2 * n - 1) * z * W[n, m + 1] - (n + m - 1) * W[n - 1, m + 1]) / (n - m)

        # Evaluate spherical harmonics
        bh_A0 = 0
        bh_A1 = 0
        bh_B1 = 0
        bh_A2 = 0
        bh_B2 = 0
        bw_A0 = 0
        bw_A1 = 0
        bw_B1 = 0
        bw_A2 = 0
        bw_B2 = 0
        ch_A0 = 0
        ch_A1 = 0
        ch_B1 = 0
        ch_A2 = 0
        ch_B2 = 0
        cw_A0 = 0
        cw_A1 = 0
        cw_B1 = 0
        cw_A2 = 0
        cw_B2 = 0

        i = -1
        for n in range(0, n_max + 1):
            for m in range(0, n + 1):
                i = i + 1

                bh_A0 = bh_A0 + (anm_bh_A0[i] * V[n + 1, m + 1] + bnm_bh_A0[i] * W[n + 1, m + 1])
                bh_A1 = bh_A1 + (anm_bh_A1[i] * V[n + 1, m + 1] + bnm_bh_A1[i] * W[n + 1, m + 1])
                bh_B1 = bh_B1 + (anm_bh_B1[i] * V[n + 1, m + 1] + bnm_bh_B1[i] * W[n + 1, m + 1])
                bh_A2 = bh_A2 + (anm_bh_A2[i] * V[n + 1, m + 1] + bnm_bh_A2[i] * W[n + 1, m + 1])
                bh_B2 = bh_B2 + (anm_bh_B2[i] * V[n + 1, m + 1] + bnm_bh_B2[i] * W[n + 1, m + 1])

                bw_A0 = bw_A0 + (anm_bw_A0[i] * V[n + 1, m + 1] + bnm_bw_A0[i] * W[n + 1, m + 1])
                bw_A1 = bw_A1 + (anm_bw_A1[i] * V[n + 1, m + 1] + bnm_bw_A1[i] * W[n + 1, m + 1])
                bw_B1 = bw_B1 + (anm_bw_B1[i] * V[n + 1, m + 1] + bnm_bw_B1[i] * W[n + 1, m + 1])
                bw_A2 = bw_A2 + (anm_bw_A2[i] * V[n + 1, m + 1] + bnm_bw_A2[i] * W[n + 1, m + 1])
                bw_B2 = bw_B2 + (anm_bw_B2[i] * V[n + 1, m + 1] + bnm_bw_B2[i] * W[n + 1, m + 1])

                ch_A0 = ch_A0 + (anm_ch_A0[i] * V[n + 1, m + 1] + bnm_ch_A0[i] * W[n + 1, m + 1])
                ch_A1 = ch_A1 + (anm_ch_A1[i] * V[n + 1, m + 1] + bnm_ch_A1[i] * W[n + 1, m + 1])
                ch_B1 = ch_B1 + (anm_ch_B1[i] * V[n + 1, m + 1] + bnm_ch_B1[i] * W[n + 1, m + 1])
                ch_A2 = ch_A2 + (anm_ch_A2[i] * V[n + 1, m + 1] + bnm_ch_A2[i] * W[n + 1, m + 1])
                ch_B2 = ch_B2 + (anm_ch_B2[i] * V[n + 1, m + 1] + bnm_ch_B2[i] * W[n + 1, m + 1])

                cw_A0 = cw_A0 + (anm_cw_A0[i] * V[n + 1, m + 1] + bnm_cw_A0[i] * W[n + 1, m + 1])
                cw_A1 = cw_A1 + (anm_cw_A1[i] * V[n + 1, m + 1] + bnm_cw_A1[i] * W[n + 1, m + 1])
                cw_B1 = cw_B1 + (anm_cw_B1[i] * V[n + 1, m + 1] + bnm_cw_B1[i] * W[n + 1, m + 1])
                cw_A2 = cw_A2 + (anm_cw_A2[i] * V[n + 1, m + 1] + bnm_cw_A2[i] * W[n + 1, m + 1])
                cw_B2 = cw_B2 + (anm_cw_B2[i] * V[n + 1, m + 1] + bnm_cw_B2[i] * W[n + 1, m + 1])

        day_of_the_year_rad = acquisition_time.day_of_the_year / 365.25 * 2 * np.pi
        b_h = (
            bh_A0
            + bh_A1 * np.cos(day_of_the_year_rad)
            + bh_B1 * np.sin(day_of_the_year_rad)
            + bh_A2 * np.cos(2 * day_of_the_year_rad)
            + bh_B2 * np.sin(2 * day_of_the_year_rad)
        )
        b_w = (
            bw_A0
            + bw_A1 * np.cos(day_of_the_year_rad)
            + bw_B1 * np.sin(day_of_the_year_rad)
            + bw_A2 * np.cos(2 * day_of_the_year_rad)
            + bw_B2 * np.sin(2 * day_of_the_year_rad)
        )
        c_h = (
            ch_A0
            + ch_A1 * np.cos(day_of_the_year_rad)
            + ch_B1 * np.sin(day_of_the_year_rad)
            + ch_A2 * np.cos(2 * day_of_the_year_rad)
            + ch_B2 * np.sin(2 * day_of_the_year_rad)
        )
        c_w = (
            cw_A0
            + cw_A1 * np.cos(day_of_the_year_rad)
            + cw_B1 * np.sin(day_of_the_year_rad)
            + cw_A2 * np.cos(2 * day_of_the_year_rad)
            + cw_B2 * np.sin(2 * day_of_the_year_rad)
        )

        return b_h, b_w, c_h, c_w

    def get_tropospheric_delay(self, sat_xyz_coordinates, acquisition_time):
        """Compute tropospheric propagation delay for the current acquisition

        :param sat_xyz_coordinates: Satellite XYZ coordinates at which calibration targets are seen as numpy array of size 3xN
        :type sat_xyz_coordinates: numpy.ndarray
        :param acquisition_time: Acquisition time [UTC]
        :type acquisition_time: PreciseDateTime
        :return: Tuple: Tropospheric propagation delays, hydrostatic and wet components, for calibration targets, both as numpy arrays of size N
        :rtype: numpy.ndarray, numpy.ndarray
        """

        # Read troposphere map files
        log.debug("Read troposphere map files")
        #   Select the four maps with times enclosing acquisition one
        acquisition_hour = (
            acquisition_time.hour_of_day
            + acquisition_time.minute_of_hour / 60.0
            + acquisition_time.second_of_minute / 3600
        )
        acquisition_hour_floor = (acquisition_time.hour_of_day // 6) * 6
        acquisition_date_floor = PreciseDateTime().set_from_numeric_datetime(
            acquisition_time.year, acquisition_time.month, acquisition_time.day_of_the_month, acquisition_hour_floor
        )

        acquisition_hour_axis = np.array([-6, 0, 6, 12])
        troposphere_map_file_list = [
            self.__get_troposphere_map_file(acquisition_date_floor + hour * 60 * 60) for hour in acquisition_hour_axis
        ]
        for troposphere_map_file in troposphere_map_file_list:
            log.debug("  Troposphere map file: {}".format(troposphere_map_file))
        tropo_map_lon_axis, tropo_map_lat_axis, a_h1, a_w1, zpd_h1, zpd_w1 = self.__read_troposphere_map_file(
            troposphere_map_file_list[0]
        )
        tropo_map_lon_axis, tropo_map_lat_axis, a_h2, a_w2, zpd_h2, zpd_w2 = self.__read_troposphere_map_file(
            troposphere_map_file_list[1]
        )
        tropo_map_lon_axis, tropo_map_lat_axis, a_h3, a_w3, zpd_h3, zpd_w3 = self.__read_troposphere_map_file(
            troposphere_map_file_list[2]
        )
        tropo_map_lon_axis, tropo_map_lat_axis, a_h4, a_w4, zpd_h4, zpd_w4 = self.__read_troposphere_map_file(
            troposphere_map_file_list[3]
        )
        acquisition_hour_axis = acquisition_hour_floor + acquisition_hour_axis

        # Interpolate in latitude/longitude/time
        log.debug("Interpolate in latitude/longitude/time")
        #   Latitude/longitude interpolation
        llh_coordinates = xyz2llh(self.xyz_coordinates)
        lon = np.rad2deg(llh_coordinates[1, :])
        lat = np.rad2deg(llh_coordinates[0, :])
        height = llh_coordinates[2, :]

        griddata_points = np.vstack((tropo_map_lon_axis, tropo_map_lat_axis)).T
        griddata_xi = np.vstack((lon, lat)).T

        zpd_h1_int = griddata(griddata_points, zpd_h1, griddata_xi, self.interpolation_method)
        zpd_h2_int = griddata(griddata_points, zpd_h2, griddata_xi, self.interpolation_method)
        zpd_h3_int = griddata(griddata_points, zpd_h3, griddata_xi, self.interpolation_method)
        zpd_h4_int = griddata(griddata_points, zpd_h4, griddata_xi, self.interpolation_method)

        zpd_w1_int = griddata(griddata_points, zpd_w1, griddata_xi, self.interpolation_method)
        zpd_w2_int = griddata(griddata_points, zpd_w2, griddata_xi, self.interpolation_method)
        zpd_w3_int = griddata(griddata_points, zpd_w3, griddata_xi, self.interpolation_method)
        zpd_w4_int = griddata(griddata_points, zpd_w4, griddata_xi, self.interpolation_method)

        a_h1_int = griddata(griddata_points, a_h1, griddata_xi, self.interpolation_method)
        a_h2_int = griddata(griddata_points, a_h2, griddata_xi, self.interpolation_method)
        a_h3_int = griddata(griddata_points, a_h3, griddata_xi, self.interpolation_method)
        a_h4_int = griddata(griddata_points, a_h4, griddata_xi, self.interpolation_method)

        a_w1_int = griddata(griddata_points, a_w1, griddata_xi, self.interpolation_method)
        a_w2_int = griddata(griddata_points, a_w2, griddata_xi, self.interpolation_method)
        a_w3_int = griddata(griddata_points, a_w3, griddata_xi, self.interpolation_method)
        a_w4_int = griddata(griddata_points, a_w4, griddata_xi, self.interpolation_method)

        #   Time interpolation
        n_points = self.xyz_coordinates.shape[1]
        a_h = np.zeros(n_points)
        a_w = np.zeros(n_points)
        zpd_h = np.zeros(n_points)
        zpd_w = np.zeros(n_points)
        for p in range(n_points):
            ah_array = np.array([a_h1_int[p], a_h2_int[p], a_h3_int[p], a_h4_int[p]])
            aw_array = np.array([a_w1_int[p], a_w2_int[p], a_w3_int[p], a_w4_int[p]])
            zhd_array = np.array([zpd_h1_int[p], zpd_h2_int[p], zpd_h3_int[p], zpd_h4_int[p]])
            zwd_array = np.array([zpd_w1_int[p], zpd_w2_int[p], zpd_w3_int[p], zpd_w4_int[p]])

            a_h[p] = interp1d(acquisition_hour_axis, ah_array, self.interpolation_method)(acquisition_hour)
            a_w[p] = interp1d(acquisition_hour_axis, aw_array, self.interpolation_method)(acquisition_hour)
            zpd_h[p] = interp1d(acquisition_hour_axis, zhd_array, self.interpolation_method)(acquisition_hour)
            zpd_w[p] = interp1d(acquisition_hour_axis, zwd_array, self.interpolation_method)(acquisition_hour)

        # Compute mapping function (with different methods)
        log.debug("Compute mapping function")
        # #   Method 1: Simplified mapping function
        # incidence_angle = compute_incidence_angle(sat_xyz_coordinates, self.xyz_coordinates)
        # mapping_function_h = 1/np.cos(incidence_angle)
        # mapping_function_w = mapping_function_h

        #   Method 2: Actual VMF3 mapping function
        #     Read spherical harmonics coefficients
        #     --> path for download: https://vmf.geo.tuwien.ac.at/codes/vmf3.m
        anm_bh = self.__read_spherical_harmonics_coefficients_file(
            os.path.join(self.troposphere_maps_dir, "anm_bh_microwave.txt")
        )
        bnm_bh = self.__read_spherical_harmonics_coefficients_file(
            os.path.join(self.troposphere_maps_dir, "bnm_bh_microwave.txt")
        )
        anm_bw = self.__read_spherical_harmonics_coefficients_file(
            os.path.join(self.troposphere_maps_dir, "anm_bw_microwave.txt")
        )
        bnm_bw = self.__read_spherical_harmonics_coefficients_file(
            os.path.join(self.troposphere_maps_dir, "bnm_bw_microwave.txt")
        )
        anm_ch = self.__read_spherical_harmonics_coefficients_file(
            os.path.join(self.troposphere_maps_dir, "anm_ch_microwave.txt")
        )
        bnm_ch = self.__read_spherical_harmonics_coefficients_file(
            os.path.join(self.troposphere_maps_dir, "bnm_ch_microwave.txt")
        )
        anm_cw = self.__read_spherical_harmonics_coefficients_file(
            os.path.join(self.troposphere_maps_dir, "anm_cw_microwave.txt")
        )
        bnm_cw = self.__read_spherical_harmonics_coefficients_file(
            os.path.join(self.troposphere_maps_dir, "bnm_cw_microwave.txt")
        )

        #     Interpolate coefficients
        b_h = np.zeros(n_points)
        b_w = np.zeros(n_points)
        c_h = np.zeros(n_points)
        c_w = np.zeros(n_points)
        for p in range(n_points):
            b_h[p], b_w[p], c_h[p], c_w[p] = self.__interpolate_spherical_harmonics_coefficients(
                anm_bh, bnm_bh, anm_bw, bnm_bw, anm_ch, bnm_ch, anm_cw, bnm_cw, acquisition_time, lat[p], lon[p]
            )

        incidence_angle = compute_incidence_angle(sat_xyz_coordinates, self.xyz_coordinates)
        elevation_angle_sin = np.sin(np.pi / 2 - incidence_angle)
        mapping_function_h = (1 + (a_h / (1 + b_h / (1 + c_h)))) / (
            elevation_angle_sin + (a_h / (elevation_angle_sin + b_h / (elevation_angle_sin + c_h)))
        )
        mapping_function_w = (1 + (a_w / (1 + b_w / (1 + c_w)))) / (
            elevation_angle_sin + (a_w / (elevation_angle_sin + b_w / (elevation_angle_sin + c_w)))
        )

        # Correct path delays for delta height
        log.debug("Correct path delays for delta height")
        #   Read elevation model height
        #   --> path for download: https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_<resolution>.txt
        grid_point_coordinates_file = os.path.join(
            self.troposphere_maps_dir, "gridpoint_coord_{}.txt".format(self.troposphere_maps_resolution)
        )
        grid_lat_axis, grid_lon_axis, grid_height = self.__read_grid_point_coordinates_file(grid_point_coordinates_file)

        #   Interpolate elevation model height
        grid_height_int = griddata(
            np.vstack((grid_lon_axis, grid_lat_axis)).T, grid_height, griddata_xi, self.interpolation_method
        )

        #   Compute zenit path delays
        h_g = grid_height_int  # height of elevation model (ETOPO5) at target location [m]
        h_s = height  # target height above ellipsoid [m]
        p_h_g = (
            zpd_h / 0.0022768 * (1 - 0.00266 * np.cos(2 * np.deg2rad(lat)) - 0.28e-6 * h_g)
        )  # pressure at h_g [mbar]
        delta_p = (1013.25 * (1 - 0.0000226 * h_s) ** 5.225) - (1013.25 * (1 - 0.0000226 * h_g) ** 5.225)
        p_h_s = p_h_g + delta_p  # pressure at h_s [mbar]

        zpd_h = 0.0022768 * p_h_s / (1 - 0.00266 * np.cos(2 * np.deg2rad(lat)) - 0.28e-6 * h_s)
        zpd_w = zpd_w * np.exp(-(h_s - h_g) / 2000.0)

        # Compute tropospheric path delays in slant range [m]
        log.debug("Compute tropospheric path delay")
        tropospheric_delay_hydrostatic = zpd_h * mapping_function_h
        tropospheric_delay_wet = zpd_w * mapping_function_w

        return tropospheric_delay_hydrostatic, tropospheric_delay_wet
