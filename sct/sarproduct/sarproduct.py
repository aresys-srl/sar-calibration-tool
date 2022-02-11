# -*- coding: utf-8 -*-

"""
SAR product module
"""

import enum
import os
import numpy as np

import arepytools.constants as cst
from arepytools.math import genericpoly
from arepytools.timing.precisedatetime import PreciseDateTime


class EDataQuantity(enum.Enum):
    """EDataQuantity class"""

    beta_nought = "BETA NOUGHT"
    sigma_nought = "SIGMA NOUGHT"
    gamma_nought = "GAMMA NOUGHT"


class SARProductROI:
    """SARProductROI class"""

    def __init__(self, t_rg_0, t_rg_step, n_rg, t_rg_end, t_az_0, t_az_step, n_az, t_az_end, rg_step, az_step, tags):
        """Initialise SARProductROI object

        :param t_rg_0: SAR product range start times [s] as numpy array of size Nchannels x Nbursts
        :type t_rg_0: numpy.ndarray
        :param t_rg_step: SAR product range steps [s] as numpy array of size Nchannels x Nbursts
        :type t_rg_step: numpy.ndarray
        :param n_rg: SAR product range samples [] as numpy array of size Nchannels x Nbursts
        :type n_rg: numpy.ndarray
        :param t_rg_end: SAR product range end times [s] as numpy array of size Nchannels x Nbursts
        :type t_rg_end: numpy.ndarray
        :param t_az_0: SAR product azimuth start times [UTC] as numpy array of size Nchannels x Nbursts
        :type t_az_0: numpy.ndarray
        :param t_az_step: SAR product azimuth steps [s] as numpy array of size Nchannels x Nbursts
        :type t_az_step: numpy.ndarray
        :param n_az: SAR product azimuth lines [] as numpy array of size Nchannels x Nbursts
        :type n_az: numpy.ndarray
        :param t_az_end: SAR product azimuth end times [UTC] as numpy array of size Nchannels x Nbursts
        :type t_az_end: numpy.ndarray
        :param rg_step: SAR product range steps [m] as numpy array of size Nchannels x Nbursts
        :type rg_step: numpy.ndarray
        :param az_step: SAR product azimuth steps [m] as numpy array of size Nchannels x Nbursts
        :type az_step: numpy.ndarray
        :param tags: SAR product tags (swath, polarization, burst) as numpy array of size Nchannels x Nbursts
        :type tags: numpy.ndarray
        """

        self.t_rg_0 = t_rg_0
        self.t_rg_step = t_rg_step
        self.n_rg = n_rg
        self.t_rg_end = t_rg_end
        self.t_az_0 = t_az_0
        self.t_az_step = t_az_step
        self.n_az = n_az
        self.t_az_end = t_az_end
        self.rg_step = rg_step
        self.az_step = az_step
        self.tags = tags

    def get_product_roi(
        self,
    ):
        """Get SAR product ROI

        :return: Tuple: Parameters describing SAR product ROI
        :rtype: numpy.ndarray x 11
        """

        return (
            self.t_rg_0,
            self.t_rg_step,
            self.n_rg,
            self.t_rg_end,
            self.t_az_0,
            self.t_az_step,
            self.n_az,
            self.t_az_end,
            self.rg_step,
            self.az_step,
            self.tags,
        )


class SARProduct:
    """SARProduct class"""

    def __init__(self, sar_product_dir, orbit_file=None):
        """Initialise SARProduct object

        :param sar_product_dir: Path to SAR product folder
        :type sar_product_dir: str
        :param orbit_file: Path to orbit file, defaults to None
        :type orbit_file: str, optional
        """

        self.sar_product_dir = sar_product_dir
        self.orbit_file = orbit_file

        self.name = None
        self.mission = None
        self.acquisition_mode = None
        self.type = None
        self.polarization = None
        self.start_time = None
        self.stop_time = None
        self.mid_time = None
        self.orbit_number = None
        self.orbit_direction = None
        self.orbit_type = None
        self.footprint = None
        self.data_quantity = None
        self.roi = None

        self.raster_info_list = []
        self.burst_info_list = []
        self.dataset_info = []
        self.swath_info_list = []
        self.sampling_constants_list = []
        self.acquisition_timeline_list = []
        self.general_sar_orbit = []
        self.dc_vector_list = []
        self.dr_vector_list = []
        self.pulse_list = []

    def _set_product_roi(
        self,
    ):
        """Set SAR product ROI

        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        # Read useful metadata
        channels = len(self.raster_info_list)
        side_looking = self.dataset_info[0].side_looking.value
        burstwise_acquisition_flag = not (
            (self.acquisition_mode == "SM")
            or (self.acquisition_mode == "WV")
            or ((self.acquisition_mode in ["IW", "EW"]) and (self.type == "GRD"))
        )
        if not burstwise_acquisition_flag:
            bursts_ref = 1
        else:
            bursts_ref = max([burst_info.get_number_of_bursts() for burst_info in self.burst_info_list])

        # Read ROI
        t_rg_0 = np.zeros((channels, bursts_ref), dtype=float)
        t_rg_step = np.zeros((channels, bursts_ref), dtype=float)
        t_rg_end = np.zeros((channels, bursts_ref), dtype=float)
        n_rg = np.zeros((channels, bursts_ref), dtype=int)
        t_az_0 = np.empty((channels, bursts_ref), dtype=object)
        t_az_step = np.zeros((channels, bursts_ref), dtype=float)
        n_az = np.zeros((channels, bursts_ref), dtype=int)
        t_az_end = np.empty((channels, bursts_ref), dtype=object)
        tags = np.empty((channels, bursts_ref), dtype=object)
        if not burstwise_acquisition_flag:

            for t in range(channels):
                raster_info = self.raster_info_list[t]
                swath_info = self.swath_info_list[t]
                t_rg_0[t, 0] = raster_info.samples_start
                t_rg_step[t, 0] = raster_info.samples_step
                n_rg[t, 0] = raster_info.samples
                t_rg_end[t, 0] = t_rg_0[t, 0] + (n_rg[t, 0] - 1) * t_rg_step[t, 0]
                t_az_0[t, 0] = raster_info.lines_start
                t_az_step[t, 0] = raster_info.lines_step
                n_az[t, 0] = raster_info.lines
                t_az_end[t, 0] = t_az_0[t, 0] + (n_az[t, 0] - 1) * t_az_step[t, 0]
                tags[t, 0] = (swath_info.swath, swath_info.polarization.value, 0)  # (swath, polarization, burst=0)

        else:

            for t in range(channels):
                raster_info = self.raster_info_list[t]
                burst_info = self.burst_info_list[t]
                swath_info = self.swath_info_list[t]
                bursts = burst_info.get_number_of_bursts()
                for tt in range(bursts):
                    t_rg_0[t, tt] = burst_info.get_range_start_time(tt)
                    t_rg_step[t, tt] = raster_info.samples_step
                    n_rg[t, tt] = raster_info.samples
                    t_rg_end[t, tt] = t_rg_0[t, tt] + (n_rg[t, tt] - 1) * t_rg_step[t, tt]
                    t_az_0[t, tt] = burst_info.get_azimuth_start_time(tt)
                    t_az_step[t, tt] = raster_info.lines_step
                    n_az[t, tt] = burst_info.lines_per_burst
                    t_az_end[t, tt] = t_az_0[t, tt] + (n_az[t, tt] - 1) * t_az_step[t, tt]
                    tags[t, tt] = (swath_info.swath, swath_info.polarization.value, tt)  # (swath, polarization, burst)
            t_az_0[t_az_0 == None] = PreciseDateTime()
            t_az_end[t_az_end == None] = PreciseDateTime()
            tags[tags == None] = ""

        # Derive additional useful information: pixel spacing
        rg_step = np.zeros(t_rg_0.shape)
        az_step = np.zeros(t_rg_0.shape)
        for t in range(t_rg_0.shape[0]):
            for tt in range(t_rg_0.shape[1]):

                if n_rg[t, tt]:
                    if self.type == "SLC":
                        t_az_mid = t_az_0[t, tt] + n_az[t, tt] / 2 * t_az_step[t, tt]
                        t_rg_mid = t_rg_0[t, tt] + n_rg[t, tt] / 2 * t_rg_step[t, tt]
                        look_angle = (
                            self.general_sar_orbit[0].get_look_angle(t_az_mid, t_rg_mid, side_looking) / 180 * np.pi
                        )
                        vg = self.general_sar_orbit[0].get_velocity_ground(t_az_mid, look_angle)
                        rg_step[t, tt] = t_rg_step[t, tt] * cst.LIGHT_SPEED / 2
                        az_step[t, tt] = t_az_step[t, tt] * vg

                    elif self.type == "GRD":
                        raise NotImplementedError  # TODO
                        """
                        Coefficients = PFGround2SlantObj_Ref.get_poly(0).coefficients
                        if Coefficients[0] > 1:  # GroundToSlant polynomials expressed in meters
                            ConversionFactor = cst.LIGHT_SPEED/2
                        else:  # GroundToSlant polynomials expressed in seconds
                            ConversionFactor = 1
                        """
                        t_az_mid = t_az_0[t, tt] + n_az[t, tt] / 2 * t_az_step[t, tt]
                        t_rg_mid = t_rg_0[t, tt] + n_rg[t, tt] / 2 * t_rg_step[t, tt]
                        """
                        t_rg_mid = gp.create_sorted_poly_list(PFGround2SlantObj_Ref).evaluate((t_az_mid, t_rg_mid)) / ConversionFactor
                        """
                        look_angle = (
                            self.general_sar_orbit[0].get_look_angle(t_az_mid, t_rg_mid, side_looking) / 180 * np.pi
                        )
                        vg = self.general_sar_orbit[0].get_velocity_ground(t_az_mid, look_angle)
                        rg_step[t, tt] = t_rg_step[t, tt]
                        az_step[t, tt] = t_az_step[t, tt] * vg

        # Set ROI
        self.roi = SARProductROI(
            t_rg_0, t_rg_step, n_rg, t_rg_end, t_az_0, t_az_step, n_az, t_az_end, rg_step, az_step, tags
        )

        return True

    def get_swath_index(self, swath):
        """Get swath index

        :param swath: Swath name
        :type swath: str
        :return: Swath index
        :rtype: int
        """

        swath_index = [t[0] for t in self.roi.tags[:, 0]].index(swath)

        return swath_index

    def get_range_time_from_position(self, swath, burst, position):
        """Get range time from position

        :param swath: Swath name or index
        :type swath: str / int
        :param burst: Burst number
        :type burst: int
        :param position: Position in swath and burst []
        :type position: float
        :return: Range time [s]
        :rtype: float
        """

        if type(swath) is int:
            swath_index = swath
        else:
            swath_index = self.get_swath_index(swath)

        t_rg = self.roi.t_rg_0[swath_index, burst] + position * self.roi.t_rg_step[swath_index, burst]

        return t_rg

    def get_azimuth_time_from_position(self, swath, burst, position):
        """Get azimuth time from position

        :param swath: Swath name or index
        :type swath: str / int
        :param burst: Burst number
        :type burst: int
        :param position: Position in swath and burst []
        :type position: float
        :return: Azimuth time [UTC]
        :rtype: PreciseDateTime
        """

        if type(swath) is int:
            swath_index = swath
        else:
            swath_index = self.get_swath_index(swath)

        t_az = (
            self.roi.t_az_0[swath_index, burst]
            + (position - np.sum(self.roi.n_az[swath_index, 0:burst])) * self.roi.t_az_step[swath_index, burst]
        )

        return t_az

    def get_squint(self, swath, burst, t_rg, t_az):
        """Get squint angle for a given position

        :param swath: Swath name or index
        :type swath: str / int
        :param burst: Burst number
        :type burst: int
        :param t_rg: Range time [s]
        :type t_rg: float
        :param t_az: Azimuth time [UTC]
        :type t_az: PreciseDateTime
        :return: Tuple: Squint angle [deg] and frequency [Hz]
        :rtype: float, float
        """

        if type(swath) is int:
            swath_index = swath
        else:
            swath_index = self.get_swath_index(swath)

        fc_hz = self.dataset_info[0].fc_hz
        vel = np.linalg.norm(self.general_sar_orbit[0].get_velocity(t_az))

        dc = genericpoly.create_sorted_poly_list(self.dc_vector_list[swath_index]).evaluate((t_az, t_rg))
        fr = genericpoly.create_sorted_poly_list(self.dr_vector_list[swath_index]).evaluate((t_az, t_rg))

        azimuth_steering_rate = self.swath_info_list[swath_index].azimuth_steering_rate_pol
        if azimuth_steering_rate[0] > 0:  # TOPSAR case
            ar = 2 * vel / (cst.LIGHT_SPEED / fc_hz) * azimuth_steering_rate[0]
            amr = -fr * ar / (ar - fr)
            squint_frequency = (
                amr
                * (
                    t_az
                    - (
                        self.roi.t_az_0[swath_index, burst]
                        + (self.roi.n_az[swath_index, burst] - 1) / 2 * self.roi.t_az_step[swath_index, burst]
                    )
                )
                + dc
            )
        elif azimuth_steering_rate[0] < 0:  # SPOT case
            squint_frequency = dc  # TODO To be reviewed
        else:  # STRIPMAP and SCANSAR cases
            squint_frequency = dc

        squint_angle = squint_frequency / (2 * vel / (cst.LIGHT_SPEED / fc_hz)) * 180 / np.pi

        return squint_angle, squint_frequency

    def get_doppler_rate_theoretical(self, pt_geo, t_az=None):
        """Compute theoretical Doppler rate

        :param pt_geo: XYZ coordinates as numpy array of size 3xN
        :type pt_geo: numpy.ndarray
        :param t_az: Satellite XYZ coordinates as numpy array of size 3xN, defaults to None
        :type t_az: numpy.ndarray, optional
        :return: Theoretical Doppler rate as numpy array of size N
        :rtype: numpy.ndarray
        """

        if t_az is None:
            raise NotImplementedError  # TODO

        fc_hz = self.dataset_info[0].fc_hz

        p_sat = self.general_sar_orbit[0].get_position(t_az)
        v_sat = self.general_sar_orbit[0].get_velocity(t_az)
        a_sat = self.general_sar_orbit[0].get_acceleration(t_az)

        r = (p_sat - pt_geo).transpose()
        r_norm = np.linalg.norm(r)

        doppler_rate_theoretical = (
            -2
            / (cst.LIGHT_SPEED / fc_hz)
            / r_norm
            * (np.linalg.norm(v_sat) ** 2 + float(np.dot(r, a_sat)) - (float(np.dot(r, v_sat)) / r_norm) ** 2)
        )

        return doppler_rate_theoretical

    def convert_coordinates_geo2sar(self, pt_geo, pt_rg_delay=None):
        """Convert XYZ coordinates to SAR coordinates

        :param pt_geo: XYZ coordinates as numpy array of size 3xN
        :type pt_geo: numpy.ndarray
        :param pt_rg_delay: Range delays [s] as numpy array of size N, defaults to None
        :type pt_rg_delay: numpy.ndarray, optional
        :return: Tuple: Range and azimuth SAR coordinates [s and UTC], both as numpy arrays of size N
        :rtype: numpy.ndarray, numpy.ndarray
        """

        # Read useful metadata
        fc_hz = self.dataset_info[0].fc_hz

        # Convert coordinates
        targets = pt_geo.shape[1]
        pt_sar__t_rg = np.zeros((1, targets), dtype=float)
        pt_sar__t_az = np.empty((1, targets), dtype=object)
        if pt_rg_delay is None:
            pt_rg_delay = np.zeros((1, targets), dtype=float)
        for p in range(targets):
            try:
                t_az, t_rg = self.general_sar_orbit[0].earth2sat(
                    pt_geo[:, p], 0.0, cst.LIGHT_SPEED / fc_hz
                )  # monostatic geocoding
                # t_az, t_rg = self.general_sar_orbit[0].earth2sat(pt_geo[:,p], 0., cst.LIGHT_SPEED/fc_hz, PFGSOObj_Ref)   # bistatic geocoding
                t_az = t_az[0]
                t_rg = t_rg[0]
            except RuntimeError:
                t_az = PreciseDateTime()
                t_rg = 0
            t_rg = t_rg + pt_rg_delay[0, p]

            if self.type == "SLC":
                pt_sar__t_rg[0, p] = t_rg
                pt_sar__t_az[0, p] = t_az
            elif self.type == "GRD":
                raise NotImplementedError  # TODO
                """
                if not PFSlant2GroundObj_Ref:
                    # PFSlant2GroundObj_Ref = convertPolynomialsG2S2S2G(PFGround2SlantObj_Ref,
                    #                                                   PFRasterInfoObj_Ref.samples_start,
                    #                                                   PFRasterInfoObj_Ref.samples_step,
                    #                                                   PFRasterInfoObj_Ref.samples)
                # PolyArray = getPolyArray(PFSlant2GroundObj_Ref)
                # RefY = getRefY(PolyArray(1))
                _, RefY = gp._create_generic_poly(PFSlant2GroundObj_Ref.get_poly(0)).reference_values
                if RefY > 1:  # SlantToGround polynomials expressed in meters
                    ConversionFactor = LightSpeed / 2
                else:  # SlantToGround polynomials expressed in seconds
                    ConversionFactor = 1
                pt_sar__t_rg[p] = gp._create_generic_poly(PFSlant2GroundObj_Ref.get_poly(0)).evaluate((t_az, t_rg * ConversionFactor))
                pt_sar__t_az[p] = t_az
                """

        return pt_sar__t_rg, pt_sar__t_az

    def convert_coordinates_sar2roi(self, pt_sar__t_rg, pt_sar__t_az):
        """Convert SAR coordinates to ROI coordinates

        :param pt_sar__t_rg: Range SAR coordinates [s] as numpy array of size N
        :type pt_sar__t_rg: numpy.ndarray
        :param pt_sar__t_az: Azimuth SAR coordinates [UTC] as numpy array of size N
        :type pt_sar__t_az: numpy.ndarray
        :return: Tuple: Range and azimuth masks of ROI coordinates, both as numpy arrays of size Nchannels x Nbursts x Ntargets
        :rtype: numpy.ndarray, numpy.ndarray
        """

        targets = pt_sar__t_rg.shape[1]
        pt_sar__rg_pos__mask = np.zeros((self.roi.t_rg_0.shape[0], self.roi.t_rg_0.shape[1], targets), dtype=float)
        pt_sar__az_pos__mask = np.zeros((self.roi.t_rg_0.shape[0], self.roi.t_rg_0.shape[1], targets), dtype=float)
        for p in range(targets):
            for t in range(self.roi.t_rg_0.shape[0]):
                for tt in range(self.roi.t_rg_0.shape[1]):
                    if self.roi.n_rg[t, tt] and self.roi.n_az[t, tt]:
                        if (
                            self.roi.t_rg_0[t, tt] <= pt_sar__t_rg[0, p] <= self.roi.t_rg_end[t, tt]
                            and self.roi.t_az_0[t, tt] <= pt_sar__t_az[0, p] <= self.roi.t_az_end[t, tt]
                        ):
                            pt_sar__rg_pos__mask[t, tt, p] = (
                                pt_sar__t_rg[0, p] - self.roi.t_rg_0[t, tt]
                            ) / self.roi.t_rg_step[t, tt]
                            pt_sar__az_pos__mask[t, tt, p] = (
                                pt_sar__t_az[0, p] - self.roi.t_az_0[t, tt]
                            ) / self.roi.t_az_step[t, tt] + sum(self.roi.n_az[t, 0:tt])

        return pt_sar__rg_pos__mask, pt_sar__az_pos__mask

    def select_data_portion(self, data_portion_rect):
        """Select a given data portion in SAR product ROI

        :param data_portion_rect: Data portion coordinates as numpy array of size 4 (first azimuth, first range, number of lines, number of samples)
        :type data_portion_rect: numpy.ndarray
        :return: Tuple: Data portion corners range and azimuth ROI coordinates, both as numpy arrays of size Nchannels x Nbursts x 2
        :rtype: numpy.ndarray, numpy.ndarray
        """

        # Compute relative azimuth times
        t_az_0__rel = self.roi.t_az_0 - self.roi.t_az_0[0, 0]
        t_az_end__rel = self.roi.t_az_end - self.roi.t_az_0[0, 0]

        # Set boundaries
        ind = self.roi.n_rg > 0
        axes_limits = np.array(
            [
                np.min(t_az_0__rel[ind]),
                np.max(t_az_end__rel[ind]),
                np.min(self.roi.t_rg_0[ind]),
                np.max(self.roi.t_rg_end[ind]),
            ]
        )

        # Reorganize inputs
        data_portion_corners_az__axes = np.sort([data_portion_rect[0], data_portion_rect[0] + data_portion_rect[2]])
        data_portion_corners_rg__axes = np.sort([data_portion_rect[1], data_portion_rect[1] + data_portion_rect[3]])

        # Check data portion boundaries
        # - if data portion is outside axes, return empty variables
        if (
            (data_portion_corners_az__axes[1] < axes_limits[0])
            or (data_portion_corners_az__axes[0] > axes_limits[1])
            or (data_portion_corners_rg__axes[1] < axes_limits[2])
            or (data_portion_corners_rg__axes[0] > axes_limits[3])
        ):
            data_portion_corners_rg__samples = np.array([])
            data_portion_corners_az__lines = np.array([])
            return data_portion_corners_rg__samples, data_portion_corners_az__lines

        # - if data portion is partially outside axes / ROI, limit it to axes / ROI
        data_portion_corners_az__axes[0] = np.max(
            [data_portion_corners_az__axes[0], axes_limits[0], np.min(t_az_0__rel[:, 0])]
        )
        data_portion_corners_az__axes[1] = np.min(
            [data_portion_corners_az__axes[1], axes_limits[1], np.max(t_az_end__rel[:, -1])]
        )
        data_portion_corners_rg__axes[0] = np.max(
            [data_portion_corners_rg__axes[0], axes_limits[2], np.min(self.roi.t_rg_0[:, 0])]
        )
        data_portion_corners_rg__axes[1] = np.min(
            [data_portion_corners_rg__axes[1], axes_limits[3], np.max(self.roi.t_rg_end[:, 0])]
        )

        # - if data portion is a single point, return error
        if (data_portion_rect[2] == 0) or (data_portion_rect[3] == 0):
            raise RuntimeError("Invalid area selected.")

        # Translate data portion boundaries from original quantities into samples / lines using provided ROI
        data_portion_corners_rg__axes__curr = np.zeros((self.roi.t_rg_0.shape[0], self.roi.t_rg_0.shape[1], 2))
        data_portion_corners_az__axes__curr = np.zeros((self.roi.t_rg_0.shape[0], self.roi.t_rg_0.shape[1], 2))
        data_portion_corners_rg__samples = np.zeros((self.roi.t_rg_0.shape[0], self.roi.t_rg_0.shape[1], 2), dtype=int)
        data_portion_corners_az__lines = np.zeros((self.roi.t_rg_0.shape[0], self.roi.t_rg_0.shape[1], 2), dtype=int)
        ValidDataPortion_Flag = 0
        for t in range(data_portion_corners_rg__samples.shape[0]):
            for tt in range(data_portion_corners_rg__samples.shape[1]):
                data_portion_corners_rg__axes__curr[t, tt, 0] = max(
                    [data_portion_corners_rg__axes[0], self.roi.t_rg_0[t, tt]]
                )
                data_portion_corners_rg__axes__curr[t, tt, 1] = min(
                    [data_portion_corners_rg__axes[1], self.roi.t_rg_end[t, tt]]
                )
                data_portion_corners_az__axes__curr[t, tt, 0] = max(
                    [data_portion_corners_az__axes[0], t_az_0__rel[t, tt]]
                )
                data_portion_corners_az__axes__curr[t, tt, 1] = min(
                    [data_portion_corners_az__axes[1], t_az_end__rel[t, tt]]
                )
                if (data_portion_corners_rg__axes__curr[t, tt, 1] <= data_portion_corners_rg__axes__curr[t, tt, 0]) or (
                    data_portion_corners_az__axes__curr[t, tt, 1] <= data_portion_corners_az__axes__curr[t, tt, 0]
                ):
                    continue
                data_portion_corners_rg__samples[t, tt, :] = np.rint(
                    (data_portion_corners_rg__axes__curr[t, tt, :] - self.roi.t_rg_0[t, tt]) / self.roi.t_rg_step[t, tt]
                )
                data_portion_corners_az__lines[t, tt, :] = (
                    np.rint(
                        (data_portion_corners_az__axes__curr[t, tt, :] - t_az_0__rel[t, tt]) / self.roi.t_az_step[t, tt]
                    )
                    + tt * self.roi.n_az[t, tt]
                )
                ValidDataPortion_Flag = 1
        if not ValidDataPortion_Flag:
            raise RuntimeError("Invalid area selected.")

        return data_portion_corners_rg__samples, data_portion_corners_az__lines


def get_sar_product_mission(sar_product_dir):
    """Get SAR product mission, basing on its filename

    :param sar_product_dir: Path to SAR product folder
    :type sar_product_dir: str
    :return: SAR product mission
    :rtype: str
    """

    sar_product_name = os.path.basename(sar_product_dir)

    sar_product_mission_id = sar_product_name[0:3]
    if sar_product_mission_id == "S1A" or sar_product_mission_id == "S1B":
        sar_product_mission = "SENTINEL-1"
    elif sar_product_mission_id == "BIO":
        sar_product_mission = "BIOMASS"
    else:
        raise ValueError("SAR product mission not recognized.")

    return sar_product_mission
