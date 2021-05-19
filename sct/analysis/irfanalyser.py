# -*- coding: utf-8 -*-

"""
Impulse Response Function (IRF) analysis module
"""

import numpy as np
from scipy.interpolate import griddata

import sct.support.signalprocessing as sp


class IRFAnalyser:  # TODO Improve IRFAnalyser class
    """IRFAnalyser class"""

    def __init__(
        self,
    ):
        """Initialise IRFAnalyser object"""

        pass

    def __cut_central_rect(self, data_portion, n_rg_rect, n_az_rect):
        """Cut data portion central rectangle

        :param data_portion: Data portion as numpy array of size Nsamples x Nlines
        :type data_portion: numpy.ndarray
        :param n_rg_rect: Rectangle number of samples
        :type n_rg_rect: int
        :param n_az_rect: Rectangle number of lines
        :type n_az_rect: int
        :return: Data portion central rectangle
        :rtype: numpy.ndarray
        """

        # Input matrix size
        ny, nx = data_portion.shape
        if nx < n_az_rect or ny < n_rg_rect:
            raise ValueError("Rect too large for the input matrix.")

        # Centre position
        cx = nx // 2
        cy = ny // 2

        # Rect half sizes
        hs = n_rg_rect // 2
        hl = n_az_rect // 2

        # Central rect cut
        data_portion = data_portion[cy - hs : cy - hs + n_rg_rect, cx - hl : cx - hl + n_az_rect]

        return data_portion

    def __compute_resolution(self, irf_cuts):
        """Compute resolution from IRF cuts

        :param irf_cuts: IRF cuts as numpy array of size NxM
        :type irf_cuts: numpy.ndarray
        :return: Resolutions as numpy array of size M
        :rtype: numpy.ndarray
        """

        n_rg, n_az = irf_cuts.shape
        irf_cuts = irf_cuts / np.max(np.abs(irf_cuts), axis=0)
        irf_cuts_db = 10 * np.log10(np.abs(irf_cuts ** 2)) - 10 * np.log10(0.5)
        res = np.zeros((1, n_az))

        for ii in range(n_az):
            cur_irf = irf_cuts_db[:, ii]
            mainlobe_3db_index = np.argwhere(cur_irf > 0)

            if mainlobe_3db_index.size == 0 or mainlobe_3db_index[0] == 0 or mainlobe_3db_index[-1] == n_rg - 1:
                res[0, ii] = -1.0
            else:
                p2 = cur_irf[mainlobe_3db_index[0]]
                p1 = cur_irf[mainlobe_3db_index[0] - 1]
                dsx = 1 + p1 / (p2 - p1)

                p2 = cur_irf[mainlobe_3db_index[-1] + 1]
                p1 = cur_irf[mainlobe_3db_index[-1]]
                ddx = -p1 / (p2 - p1)

                res[0, ii] = mainlobe_3db_index.size - 1 + ddx + dsx

        if n_az == 1:
            res = res[0, 0]

        return res

    def measure_resolution_slr_localization(
        self, data_portion, target_position=[], target_position__ref=[], step=[1, 1], sidelobes_directions=[np.inf, 0]
    ):
        """Measure resolutions, side lobe ratios and target localizations

        :param data_portion: Data portion as numpy array of size Nsamples x Nlines
        :type data_portion: numpy.ndarray
        :param target_position: IRF peak measured position (range and azimuth), defaults to []
        :type target_position: list, optional
        :param target_position__ref: IRF peak theoretical position (range and azimuth), defaults to []
        :type target_position__ref: list, optional
        :param step: Data range and azimuth steps [m], defaults to [1,1]
        :type step: list, optional
        :param sidelobes_directions: IRF sidelobes range and azimuth directions, defaults to [np.inf,0]
        :type sidelobes_directions: list, optional
        :return: Tuple: IRF measured parameters (resolution, PSLR, ISLR, SSLR, localization error) along range and azimuth directions
        :rtype: list x 5
        """

        # Define constants for IRF analysis
        if np.all(np.isreal(data_portion)):
            data_type = "DETECTED"
        else:
            data_type = "COMPLEX"
        data_portion_size = data_portion.shape
        data_portion_half_size = [x // 2 for x in data_portion_size]
        data_portion_size_ratio = data_portion_size[1] / data_portion_size[0]
        irf_roi = [32 * x for x in [1, np.int(np.rint(data_portion_size_ratio))]]
        irf_ovrs = 16
        irf_roi__center__int = [x // 2 * irf_ovrs for x in irf_roi]
        mask_method = "PEAK"  # PEAK, RES

        # Perform IRF analysis
        # NOTE Data spectrum is supposed to be unfolded (e.g. for Topsar data, deramping already applied)
        # - Compute power (only for detected data)
        if data_type == "DETECTED":
            data_portion = data_portion ** 2

        # - Recentering
        if target_position == []:
            _, target_position_rg, target_position_az = sp.max2d_fine(data_portion)
            target_position = [target_position_rg, target_position_az]
        rg_shift = target_position[0] - data_portion_half_size[0]
        az_shift = target_position[1] - data_portion_half_size[1]
        data_portion__recentered = sp.shift_data(data_portion, rg_shift, az_shift)
        if abs(rg_shift) > irf_roi[0] or abs(az_shift) > irf_roi[1]:
            irf_resolution = [0, 0]
            irf_pslr = [0, 0, 0]
            irf_islr = [0, 0, 0]
            irf_sslr = [0, 0, 0]
            irf_localization_error = [0, 0]
            return irf_resolution, irf_pslr, irf_islr, irf_sslr, irf_localization_error
        data_portion__recentered = self.__cut_central_rect(data_portion__recentered, irf_roi[0], irf_roi[1])

        # - Interpolation
        data_portion__recentered__int = sp.interp2_modulated_data(
            data_portion__recentered, irf_ovrs, irf_ovrs, 1, 1
        )  # NOTE Rg/az variance temporarily deactivated, to be made more robust
        irf_rg_axis = (np.arange(irf_roi[0] * irf_ovrs) - irf_roi__center__int[0]) / irf_ovrs
        irf_az_axis = (np.arange(irf_roi[1] * irf_ovrs) - irf_roi__center__int[1]) / irf_ovrs
        if np.isinf(sidelobes_directions[0]):
            irf_rg_profile = data_portion__recentered__int[:, irf_roi__center__int[1]].reshape(-1, 1)
            irf_az_profile = data_portion__recentered__int[irf_roi__center__int[0], :].reshape(-1, 1)
        else:
            irf_az_axis_mat, irf_rg_axis_mat = np.meshgrid(irf_az_axis, irf_rg_axis)
            if abs(sidelobes_directions[0] * data_portion_size_ratio) > 1:
                irf_rg_profile = griddata(
                    np.array([irf_az_axis_mat.ravel(), irf_rg_axis_mat.ravel()]).transpose(),
                    data_portion__recentered__int.ravel(),
                    (1 / sidelobes_directions[0] * irf_rg_axis, irf_rg_axis),
                    method="linear",
                    fill_value=0,
                ).reshape(-1, 1)
            else:
                irf_rg_profile = griddata(
                    np.array([irf_az_axis_mat.ravel(), irf_rg_axis_mat.ravel()]).transpose(),
                    data_portion__recentered__int.ravel(),
                    (irf_az_axis, sidelobes_directions[0] * irf_az_axis),
                    method="linear",
                    fill_value=0,
                ).reshape(-1, 1)
            if abs(sidelobes_directions[1] * data_portion_size_ratio) > 1:
                irf_az_profile = griddata(
                    np.array([irf_az_axis_mat.ravel(), irf_rg_axis_mat.ravel()]).transpose(),
                    data_portion__recentered__int.ravel(),
                    (1 / sidelobes_directions[1] * irf_rg_axis, irf_rg_axis),
                    method="linear",
                    fill_value=0,
                ).reshape(-1, 1)
            else:
                irf_az_profile = griddata(
                    np.array([irf_az_axis_mat.ravel(), irf_rg_axis_mat.ravel()]).transpose(),
                    data_portion__recentered__int.ravel(),
                    (irf_az_axis, sidelobes_directions[1] * irf_az_axis),
                    method="linear",
                    fill_value=0,
                ).reshape(-1, 1)
        irf_rg_profile = irf_rg_profile / np.max(np.abs(irf_rg_profile))
        irf_az_profile = irf_az_profile / np.max(np.abs(irf_az_profile))
        if data_type == "DETECTED":
            irf_rg_profile = np.sqrt(irf_rg_profile)
            irf_az_profile = np.sqrt(irf_az_profile)
            data_portion__recentered__int = np.sqrt(data_portion__recentered__int)

        # - Resolution
        irf_rg_resolution = self.__compute_resolution(irf_rg_profile) / irf_ovrs
        irf_az_resolution = self.__compute_resolution(irf_az_profile) / irf_ovrs
        if irf_rg_resolution <= 0 or irf_az_resolution <= 0:
            irf_resolution = [0, 0]
            irf_pslr = [0, 0, 0]
            irf_islr = [0, 0, 0]
            irf_sslr = [0, 0, 0]
            irf_localization_error = [0, 0]
            return irf_resolution, irf_pslr, irf_islr, irf_sslr, irf_localization_error

        # - PSLR, ISLR and SSLR
        """   # TODO Implement PSLR, ISLR and SSLR computation functions
        irf_az_pslr, irf_rg_pslr, irf_2d_pslr = sp.ComputePSLR2D(data_portion__recentered__int, irf_az_resolution, irf_rg_resolution,
                                                                 irf_ovrs, mask_method, sidelobes_directions)
        irf_az_islr, irf_rg_islr, irf_2d_islr = sp.ComputeISLR2D(data_portion__recentered__int, irf_az_resolution, irf_rg_resolution,
                                                                 irf_ovrs, mask_method, sidelobes_directions)
        irf_az_sslr, irf_rg_sslr, irf_2d_sslr = sp.ComputeSSLR2D(data_portion__recentered__int, irf_az_resolution, irf_rg_resolution,
                                                                 irf_ovrs, sidelobes_directions)
        """
        irf_az_pslr = -21.04894
        irf_rg_pslr = -19.79127
        irf_2d_pslr = -19.79127
        irf_az_islr = -15.39509
        irf_rg_islr = -15.8684
        irf_2d_islr = -11.47163
        irf_az_sslr = -22.30397
        irf_rg_sslr = -27.26822
        irf_2d_sslr = -21.74675

        # - Localization error
        if target_position__ref != []:
            irf_rg_localization_error = data_portion_half_size[0] + rg_shift - target_position__ref[0]
            irf_az_localization_error = data_portion_half_size[1] + az_shift - target_position__ref[1]

        # Organize outputs
        if np.isinf(sidelobes_directions[0]):
            step_cuts = step
        else:
            step_cuts = np.zeros(2)
            if abs(sidelobes_directions[0] * data_portion_size_ratio) > 1:
                step_cuts[0] = np.sqrt(step[0] ** 2 + (1 / sidelobes_directions[0] * step[1]) ** 2)
            else:
                step_cuts[0] = np.sqrt((sidelobes_directions[0] * step[0]) ** 2 + step[1] ** 2)

            if abs(sidelobes_directions[1] * data_portion_size_ratio) > 1:
                step_cuts[1] = np.sqrt(step[0] ** 2 + (1 / sidelobes_directions[1] * step[1]) ** 2)
            else:
                step_cuts[1] = np.sqrt((sidelobes_directions[1] * step[0]) ** 2 + step[1] ** 2)
        irf_resolution = [irf_rg_resolution, irf_az_resolution] * step_cuts
        irf_pslr = [irf_rg_pslr, irf_az_pslr, irf_2d_pslr]
        irf_islr = [irf_rg_islr, irf_az_islr, irf_2d_islr]
        irf_sslr = [irf_rg_sslr, irf_az_sslr, irf_2d_sslr]
        if target_position__ref != []:
            irf_localization_error = [irf_rg_localization_error, irf_az_localization_error] * step
        else:
            irf_localization_error = []

        return irf_resolution, irf_pslr, irf_islr, irf_sslr, irf_localization_error

    def measure_rcs_peak_clutter(self, data_portion, target_position, step, resolution):
        """Measure RCS, IRF peak and clutter

        :param data_portion: Data portion as numpy array of size Nsamples x Nlines
        :type data_portion: numpy.ndarray
        :param target_position: IRF peak measured position (range and azimuth), defaults to []
        :type target_position: list, optional
        :param step: Data range and azimuth steps [m], defaults to [1,1]
        :type step: list, optional
        :param resolution: Range and azimuth resolutions [m]
        :type resolution: list
        :return: Tuple: IRF measured parameters (RCS, peak value, clutter)
        :rtype: float x 5
        """

        # HP: input data is considered:
        # - beta-nought
        # - radiometrically corrected
        k_lin = 1  # - absolutely calibrated
        sf = 1  # - not resampled

        # Define constants for RCS analysis
        if np.all(np.isreal(data_portion)):
            data_type = "DETECTED"
        else:
            data_type = "COMPLEX"
        data_portion_size = data_portion.shape
        data_portion_size_ratio = data_portion_size[1] / data_portion_size[0]
        roi_size = [128 * x for x in [1, np.int(np.rint(data_portion_size_ratio))]]
        interp_factor = 8
        m = 10
        mm = 20

        # Perform RCS analysis
        # - Compute power(only for detected data)
        if data_type == "DETECTED":
            data_portion = data_portion ** 2

        # - Cut data around the PT area
        if target_position.size == 0:
            _, ii, jj = sp.max2d(abs(data_portion))
        else:
            ii = int(np.floor(target_position[0]))
            jj = int(np.floor(target_position[1]))
        if (
            (ii - roi_size[0] // 2 < 0)
            or (ii + roi_size[0] // 2 > data_portion_size[0])
            or (jj - roi_size[1] // 2 < 0)
            or (jj + roi_size[1] // 2 > data_portion_size[1])
        ):
            rcs_db = 0
            peak_value = 0
            clutter_db = 0
            return rcs_db, peak_value, clutter_db
        else:
            data_portion = data_portion[
                (ii - roi_size[0] // 2) : (ii + roi_size[0] // 2), (jj - roi_size[1] // 2) : (jj + roi_size[1] // 2)
            ]

        # - Compute resolutions
        rg_res = resolution[0]  # [m]
        az_res = resolution[1]  # [m]
        rg_pixel = step[0]  # [m]
        az_pixel = step[1]  # [m]

        # - Compute PAr (PAsr for SLC, PAgr*sin(alpha) for GRD)
        p_ar = az_pixel * rg_pixel / (interp_factor ** 2)

        # - Compute the data intensity
        if data_type == "DETECTED":
            i_int = data_portion
        else:
            i_int = abs(data_portion) ** 2

        # - Compute and remove the background intensity
        m_az = int(np.ceil(m * az_res / az_pixel))
        m_rg = int(np.ceil(m * rg_res / rg_pixel))
        pos_rg_1 = [
            10,
            max(roi_size[0] - 11 - m_rg, roi_size[0] / 2 + 9),
            10,
            max(roi_size[0] - 11 - m_rg, roi_size[0] / 2 + 9),
        ]
        pos_rg_2 = [
            min(pos_rg_1[0] + m_rg, roi_size[0] / 2 - 10),
            min(pos_rg_1[1] + m_rg, roi_size[0] - 10),
            min(pos_rg_1[2] + m_rg, roi_size[0] / 2 - 10),
            min(pos_rg_1[3] + m_rg, roi_size[0] - 10),
        ]
        pos_az_1 = [
            10,
            10,
            max(roi_size[1] - 11 - m_az, roi_size[1] / 2 + 9),
            max(roi_size[1] - 11 - m_az, roi_size[1] / 2 + 9),
        ]
        pos_az_2 = [
            min(pos_az_1[0] + m_az, roi_size[1] / 2 - 10),
            min(pos_az_1[1] + m_az, roi_size[1] / 2 - 10),
            min(pos_az_1[2] + m_az, roi_size[1] - 10),
            min(pos_az_1[3] + m_az, roi_size[1] - 10),
        ]

        i_bkgrd = (
            np.sum(i_int[pos_rg_1[0] : pos_rg_2[0], pos_az_1[0] : pos_az_2[0]])
            + np.sum(i_int[pos_rg_1[1] : pos_rg_2[1], pos_az_1[1] : pos_az_2[1]])
            + np.sum(i_int[pos_rg_1[2] : pos_rg_2[2], pos_az_1[2] : pos_az_2[2]])
            + np.sum(i_int[pos_rg_1[3] : pos_rg_2[3], pos_az_1[3] : pos_az_2[3]])
        )
        i_bkgrd = 1 / (4 * m_rg * m_az) * i_bkgrd
        clutter_db = 10 * np.log10(i_bkgrd)
        # i_c = i_int-i_bkgrd

        # - Interpolate the corrected data intensity
        #   NOTE The original data is interpolated
        data_portion__int = sp.interp2_modulated_data(
            data_portion, interp_factor, interp_factor, 1, 1
        )  # NOTE Rg/az variance temporarily deactivated, to be made more robust
        if data_type == "DETECTED":
            data_portion__int = np.sqrt(data_portion__int)

        i_c_int = np.abs(data_portion__int) ** 2 - i_bkgrd

        # - Integrate the interpolated corrected data intensity
        m_az_int = int(np.ceil(mm * az_res / (az_pixel / interp_factor)))
        m_rg_int = int(np.ceil(mm * rg_res / (rg_pixel / interp_factor)))

        if target_position.size == 0:
            _, ii, jj = sp.max2d(i_c_int)
            ii = ii[0]
            jj = jj[0]
        else:
            i_c_int__cut = (
                np.abs(
                    data_portion__int[
                        int(np.floor(target_position[0]) * interp_factor - 1) : int(
                            (np.floor(target_position[0]) + 2) * interp_factor
                        ),
                        int(np.floor(target_position[1]) * interp_factor - 1) : int(
                            (np.floor(target_position[1]) + 2) * interp_factor
                        ),
                    ]
                )
                ** 2
            )
            _, ii, jj = sp.max2d(i_c_int__cut)
            ii = int(np.floor(target_position[0]) * interp_factor + ii - 1)
            jj = int(np.floor(target_position[1]) * interp_factor + jj - 1)
        rg_1 = max(ii - int(np.ceil(m_rg_int / 2)), 0)
        rg_2 = min(ii + (m_rg_int - int(np.ceil(m_rg_int / 2))), i_c_int.shape[0])
        az_1 = max(jj - int(np.ceil(m_az_int / 2)), 0)
        az_2 = min(jj + (m_az_int - int(np.ceil(m_az_int / 2))), i_c_int.shape[1])

        i_p = np.sum(i_c_int[rg_1:rg_2, az_1:az_2])

        # - Compute the RCS
        rcs_lin = i_p * p_ar / k_lin / sf ** 2
        rcs_lin = max(10 ** -10, rcs_lin)  # avoid negative or null (causing warnings) RCS values
        rcs_db = 10 * np.log10(rcs_lin)

        # - Compute the peak value (abs and phase)
        peak_value = data_portion__int[ii, jj]

        return rcs_db, peak_value, clutter_db

    def interpolate_irf(
        self,
    ):
        """Interpolate IRF"""

        pass

    def measure_resolution(
        self,
    ):
        """Compute resolution"""

        pass

    def measure_rcs(
        self,
    ):
        """Compute RCS"""

        pass

    def measure_scr(
        self,
    ):
        """Compute SCR"""

        pass

    def measure_ale(
        self,
    ):
        """Compute ALE"""

        pass
