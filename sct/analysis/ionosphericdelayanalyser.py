# -*- coding: utf-8 -*-

"""
Ionospheric delay analysis module
"""

import logging

log = logging.getLogger(__name__)

import numpy as np
import os
import re
from scipy.interpolate import RegularGridInterpolator

import arepytools.constants as cst
from arepytools.geometry.conversions import xyz2llh, llh2xyz

from sct.support.geometry_extended import find_ellipsoid_intersection
from sct.support.utils import compute_angle_between_vectors


class IonosphericDelayAnalyser:
    """IonosphericDelayAnalyser class"""

    def __init__(self, xyz_coordinates, ionosphere_maps_dir, ionosphere_analysis_center="cod"):
        """Initialise IonosphericDelayAnalyser object

        :param xyz_coordinates: Calibration targets XYZ coordinates as numpy array of size 3xN
        :type xyz_coordinates: numpy.ndarray
        :param ionosphere_maps_dir: Path to ionosphere maps folder
        :type ionosphere_maps_dir: str
        :param ionosphere_analysis_center: Ionosphere maps analysis center, defaults to 'cod'
        :type ionosphere_analysis_center: str, optional
        """

        self.xyz_coordinates = xyz_coordinates

        self.ionosphere_maps_dir = ionosphere_maps_dir
        self.ionosphere_analysis_center = ionosphere_analysis_center

        # Internal configuration parameters
        self.__reference_earth_radius = 6371000.0  # TODO Read parameters from ionosphere maps
        self.__ionosphere_height = 450000.0

    def __get_ionosphere_map_file(self, acquisition_time):
        """Get ionosphere map file starting from acquisition time. Returns error if file is not found

        :param acquisition_time: Acquisition time [UTC]
        :type acquisition_time: PreciseDateTime
        :return: Ionosphere map filename
        :rtype: str
        """

        """
        IONEX (ionosphere exchange) files naming convention:
            YYYY/DDD/AAAgDDD#.YYi.Z
            where:
                YYYY    4-digit year
                DDD     3-digit day of year
                AAA     Analysis center name:
                            c1p	1-day predicted solution (CODE)
                            c2p	2-day predicted solution (CODE)
                            cod	Final solution (CODE)
                            cor	Rapid solution (CODE)
                            e1p	1-day predicted solution (ESA)
                            e2p	2-day predicted solution (ESA)
                            ehr	Rapid high-rate solution, one map per hour, (ESA)
                            esa	Final solution (ESA)
                            esr	Rapid solution (ESA)
                            ilp	1-day predicted solution (IGS combined)
                            i2p	2-day predicted solution (IGS combined)
                            igr	Rapid solution (IGS combined)
                            igs	Final combined solution (IGS combined)
                            jpl	Final solution (JPL)
                            u2p	2 day predicted solution (UPC)
                            upc	Final solution (UPC)
                            uhr	Rapid high-rate solution, one map per hour, (UPC)
                            upr	Rapid solution (UPC)
                            uqr	Rapid high-rate solution, one map per 15 minutes, (UPC)    
                #       File number for the day, typically 0
                YY      2-digit year
                .Z      Unix compressed file
        """

        # Define ionosphere map filename and path
        year = acquisition_time.year
        day_of_the_year = acquisition_time.day_of_the_year

        ionosphere_analysis_center_list = [self.ionosphere_analysis_center, "cor"]
        for center in ionosphere_analysis_center_list:
            ionosphere_map_file = "{0}g{1:03}0.{2:02d}i".format(center, day_of_the_year, year % 100)
            # --> path for download: https://cddis.nasa.gov/archive/gnss/products/ionex/<year>/<day_of_the_year>/<ionosphere_map_file>
            # --> available also through FTP, without the need of using credentials
            ionosphere_map_file = os.path.join(self.ionosphere_maps_dir, ionosphere_map_file)

            # Check if exists, otherwise try alternatives
            if os.path.isfile(ionosphere_map_file):
                return ionosphere_map_file
            if os.path.isfile(ionosphere_map_file + ".Z"):
                # unzip
                # return ionosphere_map_file
                raise NotImplementedError  # TODO

        raise FileNotFoundError("No ionosphere map found for the {} date.".format(acquisition_time))

    def __parse_tec_map(self, tec_map, exponent=-1):
        """Parse ionosphere map content

        :param tec_map: Ionosphere map section content
        :type tec_map: str
        :param exponent: Power of 10 to be applied to ionosphere map data, defaults to -1
        :type exponent: int, optional
        :return: Tuple: Ionosphere map data and corresponding times, both as lists of float
        :rtype: list, list
        """

        # Parse TEC map
        tec_map_time = re.split("EPOCH OF CURRENT MAP", tec_map)[0][7:]
        y, m, d, h, mm, _ = [int(x) for x in tec_map_time.split()]
        tec_map_time = "{}-{}-{:02} {:02}:{:02}".format(y, m, d, h, mm)

        tec_map = re.split(".*END OF TEC MAP", tec_map)[0]
        tec_map_data = (
            np.stack([np.fromstring(l, sep=" ") for l in re.split(".*LAT/LON1/LON2/DLON/H\\n", tec_map)[1:]])
            * 10 ** exponent
        )

        return tec_map_time, tec_map_data

    def __read_ionosphere_map_file(self, ionosphere_map_file):
        """Read ionosphere map file

        :param ionosphere_map_file: Path to ionosphere map file
        :type ionosphere_map_file: str
        :return: Tuple: Ionosphere map data, corresponding times, latitude and longitude axes, all as numpy arrays
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        # Read ionosphere map file
        with open(ionosphere_map_file) as f:
            # Read file removing header section
            file_content = f.read()
            tec_maps = file_content.split("START OF TEC MAP")[1:]

            # Return lists of TEC maps time and data
            tec_map_hour_list = []
            tec_map_data_list = []
            for t in tec_maps:
                tec_map_time, tec_map_data = self.__parse_tec_map(t)
                tec_map_hour_list.append(int(tec_map_time[-5:-3]))
                tec_map_data_list.append(tec_map_data)
            tec_map_hour_list[-1] = 24
            tec_map_hour_list = np.asarray(tec_map_hour_list)
            tec_map_data_list = np.asarray(tec_map_data_list)

        tec_map_lat_axis = np.arange(87.5, -(87.5 + 1), -2.5)
        tec_map_lon_axis = np.arange(-180, 180 + 1, 5)

        # Make latitude axis monotonically increasing (required by interpolator)
        tec_map_data_list = tec_map_data_list[:, ::-1, :]
        tec_map_lat_axis = tec_map_lat_axis[::-1]

        return tec_map_hour_list, tec_map_data_list, tec_map_lat_axis, tec_map_lon_axis

    def get_ionospheric_delay(self, sat_xyz_coordinates, acquisition_time, fc_hz, ionospheric_delay_scaling_factor=1.0):
        """Compute ionospheric propagation delay for the current acquisition

        :param sat_xyz_coordinates: Satellite XYZ coordinates at which calibration targets are seen as numpy array of size 3xN
        :type sat_xyz_coordinates: numpy.ndarray
        :param acquisition_time: Acquisition time [UTC]
        :type acquisition_time: PreciseDateTime
        :param fc_hz: Carrier frequency [Hz]
        :type fc_hz: float
        :param ionospheric_delay_scaling_factor: Mission-dependent ionospheric propagation delay scaling factor, defaults to 1.0
        :type ionospheric_delay_scaling_factor: float, optional
        :return: Ionospheric propagation delays for calibration targets as numpy array of size N
        :rtype: numpy.ndarray
        """

        # Read ionosphere map file
        log.debug("Read ionosphere map file")
        ionosphere_map_file = self.__get_ionosphere_map_file(acquisition_time)
        log.debug("  Ionosphere map file: {}".format(ionosphere_map_file))
        tec_map_hour_list, tec_map_data_list, tec_map_lat_axis, tec_map_lon_axis = self.__read_ionosphere_map_file(
            ionosphere_map_file
        )

        # Find ionospheric pierce point (IPP)
        log.debug("Find ionospheric pierce point (IPP)")
        n_points = self.xyz_coordinates.shape[1]
        ipp_xyz_coordinates = np.zeros((3, n_points))
        ionosphere_radius = self.__reference_earth_radius + self.__ionosphere_height
        for p in range(n_points):
            sat = sat_xyz_coordinates[:, p]
            xyz = self.xyz_coordinates[:, p]
            los = xyz - sat
            _, ipp_xyz_coordinates[:, p] = find_ellipsoid_intersection(
                los, sat, semi_axis_major=ionosphere_radius, semi_axis_minor=ionosphere_radius
            )
        ipp_llh_coordinates = xyz2llh(ipp_xyz_coordinates)
        ipp_lat = np.rad2deg(ipp_llh_coordinates[0, :])
        ipp_lon = np.rad2deg(ipp_llh_coordinates[1, :])

        # Interpolate in latitude/longitude/time
        log.debug("Interpolate in latitude/longitude/time")
        #   Find TEC maps just before and after acquisition time
        acquisition_hour = (
            acquisition_time.hour_of_day
            + acquisition_time.minute_of_hour / 60.0
            + acquisition_time.second_of_minute / 3600
        )
        ind = list(filter(lambda i: i < acquisition_hour, tec_map_hour_list))[-1]
        hour1 = tec_map_hour_list[ind]
        hour2 = tec_map_hour_list[ind + 1]
        data1 = tec_map_data_list[ind, :, :]
        data2 = tec_map_data_list[ind + 1, :, :]

        #   Account for Earth rotation
        #   NOTE This will probably give wrong results for points close to either -180 or +180 longitudes
        ipp_lon1 = ipp_lon + 360.0 / 24.0 * (acquisition_hour - hour1)
        ipp_lon2 = ipp_lon + 360.0 / 24.0 * (acquisition_hour - hour2)

        #   Perform interpolation
        interpolating_function1 = RegularGridInterpolator((tec_map_lat_axis, tec_map_lon_axis), data1)
        interpolating_function2 = RegularGridInterpolator((tec_map_lat_axis, tec_map_lon_axis), data2)
        vtec = np.full(n_points, np.nan)
        for p in range(n_points):
            # Bilinear interpolation over latitude/longitude
            vtec1 = interpolating_function1((ipp_lat[p], ipp_lon1[p]))
            vtec2 = interpolating_function2((ipp_lat[p], ipp_lon2[p]))
            # Linear interpolation in time
            vtec[p] = (hour2 - acquisition_hour) / (hour2 - hour1) * vtec1 + (acquisition_hour - hour1) / (
                hour2 - hour1
            ) * vtec2

        # Compute mapping function (with different but equivalent methods)
        log.debug("Compute mapping function")
        #   Method 1: Incidence angle at IPP
        zenit_angle = compute_angle_between_vectors(ipp_xyz_coordinates, sat_xyz_coordinates - ipp_xyz_coordinates)
        mapping_function = 1 / np.cos(zenit_angle)

        # #   Method 2: Incidence angle on ground converted
        # incidence_angle = compute_incidence_angle(sat_xyz_coordinates, self.xyz_coordinates)
        # mapping_function = 1/np.sqrt(1-(self.__reference_earth_radius/(self.__reference_earth_radius+self.__ionosphere_height)*np.sin(incidence_angle))**2)

        # #   Method 3: Incidence angle on ground
        # incidence_angle = compute_incidence_angle(sat_xyz_coordinates, self.xyz_coordinates)
        # mapping_function = 1/np.cos(incidence_angle)

        # Compute ionospheric path delay in slant range [m]
        log.debug("Compute ionospheric path delay")
        ionospheric_delay = (40.3 * 1e16 / fc_hz ** 2) * vtec * mapping_function * ionospheric_delay_scaling_factor

        return ionospheric_delay
