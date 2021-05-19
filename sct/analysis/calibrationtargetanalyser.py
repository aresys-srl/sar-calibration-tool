# -*- coding: utf-8 -*-

"""
Calibration target analysis module
"""

import logging

log = logging.getLogger(__name__)

import numpy as np

import arepytools.constants as cst
from arepytools.math import genericpoly
import sct.support.signalprocessing as sp

from sct.sarproduct.sarproduct import EDataQuantity
from sct.analysis.irfanalyser import IRFAnalyser


class CalibrationTargetAnalyser:  # TODO Improve CalibrationTargetAnalyser class
    """CalibrationTargetAnalyser class"""

    def __init__(self, sar_product, calibration_target):
        """Initialise CalibrationTargetAnalyser object

        :param sar_product: SAR product object (type depends on the mission)
        :param calibration_target: Calibration target object
        :type calibration_target: CalibrationTarget
        """

        self.sar_product = sar_product
        self.calibration_target = calibration_target
        self.swath = []
        self.burst = []
        self.polarization = []
        self.position_range = []
        self.position_azimuth = []
        self.incidence_angle = []
        self.look_angle = []
        self.squint_angle = []
        self.resolution_range = []
        self.resolution_azimuth = []
        self.pslr_range = []
        self.pslr_azimuth = []
        self.pslr_2d = []
        self.islr_range = []
        self.islr_azimuth = []
        self.islr_2d = []
        self.sslr_range = []
        self.sslr_azimuth = []
        self.sslr_2d = []
        self.rcs = []
        self.clutter = []
        self.peak_error = []
        self.scr = []
        self.measured_ale_range = []
        self.measured_ale_azimuth = []
        self.n_views = 0

        # Internal configuration parameters
        self.__analyse_irf_flag = True
        self.__measure_rcs_flag = True
        self.__measure_ale_flag = True
        self.__unit_of_measure = "Meters"

    def analyse_calibration_target(self, maximum_ale=None):
        """Analyse calibration target

        :param maximum_ale: Maximum measurable ALE [m], defaults to None
        :type maximum_ale: float, optional
        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        # Check configuration parameters
        if (not self.__analyse_irf_flag) and (not self.__measure_rcs_flag) and (not self.__measure_ale_flag):
            return True

        # Read useful SAR product metadata
        (
            t_rg_0,
            t_rg_step,
            n_rg,
            _,
            t_az_0,
            t_az_step,
            n_az,
            _,
            rg_step,
            az_step,
            tags,
        ) = self.sar_product.roi.get_product_roi()
        fc_hz = self.sar_product.dataset_info[0].fc_hz
        side_looking = self.sar_product.dataset_info[0].side_looking.value

        # Read target coordinates
        if self.__measure_ale_flag:
            pt_geo = self.calibration_target.xyz
            pt_rcs = self.calibration_target.rcs
            pt_delay = np.array([[self.calibration_target.delay]])
            pt_sar__t_rg, pt_sar__t_az = self.sar_product.convert_coordinates_geo2sar(pt_geo, pt_delay)
            pt_sar__rg_pos__mask, pt_sar__az_pos__mask = self.sar_product.convert_coordinates_sar2roi(
                pt_sar__t_rg, pt_sar__t_az
            )
            pt_sar__mask = np.logical_and(pt_sar__rg_pos__mask != 0, pt_sar__az_pos__mask != 0)

        # Select data portion where to perform IRF and RCS analyses
        log.debug("Compute number of times the target is seen by the current SAR product (swath/burst combinations)")
        if sum(sum(pt_sar__mask[:, :, 0])) == 0:
            raise RuntimeError("Target outside image.")
        if maximum_ale is None:  # the ROI dimensions depend on the maximum ALE allowed
            roi_size_rg = 32
            roi_size_az = 32
        else:
            roi_size_rg = int(max(np.ceil(abs(maximum_ale) / np.mean(rg_step[:, 0]) * 2), 3))
            roi_size_az = int(max(np.ceil(abs(maximum_ale) / np.mean(az_step[:, 0]) * 2), 3))
        pt_roi = np.zeros(4)
        pt_roi[0] = (
            pt_sar__t_az[0, 0] - t_az_0[0, 0] - roi_size_az / 2 * np.mean(t_az_step[:, 0])
        )  # an average sampling steps is used
        pt_roi[1] = pt_sar__t_rg[0, 0] - roi_size_rg / 2 * np.mean(t_rg_step[:, 0])
        pt_roi[2] = roi_size_az * np.mean(t_az_step[:, 0])
        pt_roi[3] = roi_size_rg * np.mean(t_rg_step[:, 0])
        data_portion_corners_rg, data_portion_corners_az = self.sar_product.select_data_portion(pt_roi)
        if len(data_portion_corners_rg) == 0 or len(data_portion_corners_az) == 0:
            raise RuntimeError("Target outside image.")
        data_portion_corners_rg = data_portion_corners_rg * np.tile(pt_sar__mask, (1, 1, 2))
        data_portion_corners_az = data_portion_corners_az * np.tile(pt_sar__mask, (1, 1, 2))

        # Perform and display IRF and RCS analyses
        for t in range(data_portion_corners_rg.shape[0]):
            for tt in range(data_portion_corners_rg.shape[1]):
                if (
                    data_portion_corners_rg[t, tt, 0] == 0
                    and data_portion_corners_rg[t, tt, 1] == 0
                    and data_portion_corners_az[t, tt, 0] == 0
                    and data_portion_corners_az[t, tt, 1] == 0
                ):
                    continue

                log.debug("Analyse target (swath: {}, burst: {})".format(tags[t, tt][0], tt))
                irf_analyser = IRFAnalyser()

                # Read data (define square ROI around the strongest target in the selected area)
                log.debug("Read data portion around target")
                roi = [
                    data_portion_corners_az[t, tt, 0],
                    data_portion_corners_rg[t, tt, 0],
                    data_portion_corners_az[t, tt, 1] - data_portion_corners_az[t, tt, 0] + 1,
                    data_portion_corners_rg[t, tt, 1] - data_portion_corners_rg[t, tt, 0] + 1,
                ]
                data_portion = self.sar_product.read_data(t, roi)
                if sum(sum(np.abs(data_portion))) == 0:
                    continue
                if np.all(np.isreal(data_portion)):
                    _, roi_max_rg, roi_max_az = sp.max2d_fine(
                        np.abs(data_portion) ** 2
                    )  # compute power (only for detected data)
                else:
                    _, roi_max_rg, roi_max_az = sp.max2d_fine(data_portion)
                roi_size = 128
                position = [roi[1] + roi_max_rg, roi[0] + roi_max_az]
                roi = np.array(
                    [
                        roi[0] + np.floor(roi_max_az) - roi_size / 2,
                        roi[1] + np.floor(roi_max_rg) - roi_size / 2,
                        roi_size,
                        roi_size,
                    ],
                    dtype=int,
                )
                data_portion = self.sar_product.read_data(t, roi, n_rg[t, 0], sum(n_az[t, :]))
                roi_max__pos = np.array(
                    [
                        roi_size / 2 + (roi_max_rg - np.floor(roi_max_rg)),
                        roi_size / 2 + (roi_max_az - np.floor(roi_max_az)),
                    ]
                )  # the strongest target in the selected area is used for the analyses
                if (
                    np.count_nonzero(np.abs(data_portion)) / np.prod(data_portion.shape) * 100 < 50
                ):  # TODO Check threshold
                    continue

                # Derive additional useful information for the selected target
                pt_valid__flag = 1

                b_rg = self.sar_product.sampling_constants_list[t].brg_hz
                f_rg = self.sar_product.sampling_constants_list[t].frg_hz
                rg_ovrs = np.max([int(np.rint(f_rg / b_rg / 5)), 1])  # factor=5 has been chosen arbitrarily
                b_az = self.sar_product.sampling_constants_list[t].baz_hz
                f_az = self.sar_product.sampling_constants_list[t].faz_hz
                az_ovrs = np.max([int(np.rint(f_az / b_az / 5)), 1])  # factor=5 has been chosen arbitrarily

                if rg_ovrs > 1 or az_ovrs > 1:
                    roi = np.array(
                        [
                            roi[0] - roi_size * (az_ovrs - 1) / 2,
                            roi[1] - roi_size * (rg_ovrs - 1) / 2,
                            roi_size * az_ovrs,
                            roi_size * rg_ovrs,
                        ],
                        dtype=int,
                    )
                    data_portion = self.sar_product.read_data(t, roi, n_rg[t, 0], sum(n_az[t, :]))
                    roi_max__pos = np.array(
                        [roi_max__pos[0] + roi_size * (rg_ovrs - 1) / 2, roi_max__pos[1] + roi_size * (az_ovrs - 1) / 2]
                    )

                t_rg__curr = t_rg_0[t, tt] + (roi_max__pos[0] + roi[1]) * t_rg_step[t, tt]
                t_rg__near = t_rg_0[t, tt] + (roi[1]) * t_rg_step[t, tt]
                t_rg__far = t_rg_0[t, tt] + (roi[3] - 1 + roi[1]) * t_rg_step[t, tt]
                if self.sar_product.type == "GRD":
                    raise NotImplementedError  # TODO
                    """
                    coefficients = PFGround2SlantObj_Ref.coefficients
                    if coefficients[0] > 1:  # GroundToSlant polynomials expressed in meters
                        conversion_factor = LightSpeed / 2
                    else:  # GroundToSlant polynomials expressed in seconds
                        conversion_factor = 1
                    t_rg__curr = gp.create_generic_poly(PFGround2SlantObj_Ref).evaluate((TAz0[t, tt] + (NAz[t, tt] - 1) / 2 * TAzStep[t, tt], t_rg__curr)) / conversion_factor
                    t_rg__near = gp.create_generic_poly(PFGround2SlantObj_Ref).evaluate((TAz0[t, tt] + (NAz[t, tt] - 1) / 2 * TAzStep[t, tt], t_rg__near)) / conversion_factor
                    t_rg__far = gp.create_generic_poly(PFGround2SlantObj_Ref).evaluate((TAz0[t, tt] + (NAz[t, tt] - 1) / 2 * TAzStep[t, tt], t_rg__far)) / conversion_factor
                    """
                t_az__curr = t_az_0[t, tt] + (roi_max__pos[1] + roi[0] - sum(n_az[t, 0:tt])) * t_az_step[t, tt]

                incidence_angle = self.sar_product.general_sar_orbit[0].get_incidence_angle(
                    t_az__curr, t_rg__curr, side_looking
                )
                look_angle = self.sar_product.general_sar_orbit[0].get_look_angle(t_az__curr, t_rg__curr, side_looking)
                squint_angle, _ = self.sar_product.get_squint(t, tt, t_rg__curr, t_az__curr)

                dc = genericpoly.create_sorted_poly_list(self.sar_product.dc_vector_list[t]).evaluate(
                    (t_az__curr, t_rg__curr)
                )  # TODO Add electronic steering

                pt__near = self.sar_product.general_sar_orbit[0].sat2earth(
                    t_az__curr, t_rg__near, "RIGHT", 0.0, 0.0, cst.LIGHT_SPEED / fc_hz
                )
                pt__near__t_az, _ = self.sar_product.general_sar_orbit[0].earth2sat(
                    np.reshape(pt__near, (3,)), dc, cst.LIGHT_SPEED / fc_hz
                )
                pt__far = self.sar_product.general_sar_orbit[0].sat2earth(
                    t_az__curr, t_rg__far, "RIGHT", 0.0, 0.0, cst.LIGHT_SPEED / fc_hz
                )
                pt__far__t_az, _ = self.sar_product.general_sar_orbit[0].earth2sat(
                    np.reshape(pt__far, (3,)), dc, cst.LIGHT_SPEED / fc_hz
                )
                irf_rg_cut = -(roi_size - 1) / (
                    (pt__far__t_az[0] - pt__near__t_az[0]) / t_az_step[t, tt]
                )  # range cut angular coefficient in samples
                irf_az_cut = (
                    -1 / irf_rg_cut * az_step[t, tt] ** 2 / rg_step[t, tt] ** 2
                )  # azimuth cut angular coefficient in samples
                if np.abs((roi_size - 1) / irf_rg_cut) < 1:  # use vertical and horizontal cuts in case of low squint
                    irf_rg_cut = np.inf
                    irf_az_cut = 0

                # Perform IRF analysis
                log.debug("Perform IRF analysis")
                if self.__measure_ale_flag:
                    # Find target to be used as reference
                    pt_sar__pos = [
                        np.squeeze(pt_sar__rg_pos__mask[t, tt, 0]) - roi[1],
                        np.squeeze(pt_sar__az_pos__mask[t, tt, 0]) - roi[0],
                    ]
                else:
                    pt_sar__pos = np.array([])
                step = np.array([rg_step[t, tt], az_step[t, tt]])
                if self.__unit_of_measure == "Meters":
                    (
                        irf_resolution,
                        irf_pslr,
                        irf_islr,
                        irf_sslr,
                        irf_localization_error,
                    ) = irf_analyser.measure_resolution_slr_localization(
                        data_portion, roi_max__pos, pt_sar__pos, step, [irf_rg_cut, irf_az_cut]
                    )
                else:
                    (
                        irf_resolution,
                        irf_pslr,
                        irf_islr,
                        irf_sslr,
                        irf_localization_error,
                    ) = irf_analyser.measure_resolution_slr_localization(
                        data_portion, roi_max__pos, pt_sar__pos, [1, 1], [irf_rg_cut, irf_az_cut]
                    )
                if (
                    np.any(irf_resolution == 0)
                    or np.any(irf_pslr[0:2] == 0)
                    or np.any(irf_islr[0:2] == 0)
                    or np.any(irf_sslr[0:2] == 0)
                ):  # if IRF analysis has provided partially invalid results mark the target as invalid
                    pt_valid__flag = 0

                # Perform RCS analysis
                log.debug("Perform RCS analysis")
                if pt_valid__flag:  # if target has not been found during IRF analysis skip RCS analysis
                    if self.__measure_rcs_flag:
                        if self.sar_product.data_quantity == EDataQuantity.sigma_nought.value:
                            data_portion = data_portion / np.sqrt(
                                np.sin(incidence_angle / 180 * np.pi)
                            )  # sigma to beta nought conversion
                        elif self.sar_product.data_quantity == EDataQuantity.gamma_nought.value:
                            data_portion = (
                                data_portion
                                / np.sqrt(np.sin(incidence_angle / 180 * np.pi))
                                * np.sqrt(np.cos(incidence_angle / 180 * np.pi))
                            )  # gamma to beta nought conversion
                        irf_resolution__temp = irf_resolution
                        if self.sar_product.type == "GRD":
                            step[0] = step[0] * np.sin(
                                incidence_angle / 180 * np.pi
                            )  # for GRD, the pixel area is computed as PAgr*sin(alpha)
                            irf_resolution__temp[0] = irf_resolution__temp[0] * np.sin(incidence_angle / 180 * np.pi)
                        if self.__unit_of_measure == "Meters":
                            rcs, peak_value, clutter = irf_analyser.measure_rcs_peak_clutter(
                                data_portion, roi_max__pos, step, irf_resolution__temp
                            )
                        else:
                            rcs, peak_value, clutter = irf_analyser.measure_rcs_peak_clutter(
                                data_portion, roi_max__pos, step, irf_resolution__temp * step
                            )
                        if (
                            rcs == 0 or peak_value == 0 or clutter == 0
                        ):  # if RCS analysis has provided partially invalid results mark the target as invalid
                            pt_valid__flag = 0
                    else:
                        rcs = []
                        clutter = []

                # Compute RCS and peak phase errors
                log.debug("Compute RCS and peak phase errors")
                if pt_valid__flag:  # if target has not been found during IRF analysis skip errors computation
                    if self.__measure_ale_flag and self.__measure_rcs_flag:
                        polarization = tags[t, tt][1]
                        rcs_error = np.sqrt(10 ** ((rcs - pt_rcs[polarization]) / 10))
                        sat_geo = self.sar_product.general_sar_orbit[0].get_position(pt_sar__t_az[0, 0])
                        peak_phase_error = np.angle(
                            peak_value
                            * np.exp(1j * 4 * np.pi / (cst.LIGHT_SPEED / fc_hz) * np.linalg.norm(sat_geo - pt_geo))
                        )
                        peak_error = rcs_error * np.exp(1j * peak_phase_error)
                    else:
                        peak_error = []

                # Compute SCR
                log.debug("Compute SCR")
                if pt_valid__flag:  # if target has not been found during IRF analysis skip SCR computation
                    if self.__measure_rcs_flag:
                        scr = 10 * np.log10(np.abs(peak_value) ** 2) - clutter
                    else:
                        scr = []

                # If target has been marked as valid, store results
                if pt_valid__flag:
                    log.debug("Target valid, store results")
                    self.swath.append(tags[t, tt][0])
                    self.burst.append(tt)
                    self.polarization.append(tags[t, tt][1])
                    self.position_range.append(position[0])
                    self.position_azimuth.append(position[1])
                    self.incidence_angle.append(incidence_angle)
                    self.look_angle.append(look_angle)
                    self.squint_angle.append(squint_angle)
                    self.resolution_range.append(irf_resolution[0])
                    self.resolution_azimuth.append(irf_resolution[1])
                    self.pslr_range.append(irf_pslr[0])
                    self.pslr_azimuth.append(irf_pslr[1])
                    self.pslr_2d.append(irf_pslr[2])
                    self.islr_range.append(irf_islr[0])
                    self.islr_azimuth.append(irf_islr[1])
                    self.islr_2d.append(irf_islr[2])
                    self.sslr_range.append(irf_sslr[0])
                    self.sslr_azimuth.append(irf_sslr[1])
                    self.sslr_2d.append(irf_sslr[2])
                    self.rcs.append(rcs)
                    self.clutter.append(clutter)
                    self.peak_error.append(peak_error)
                    self.scr.append(scr)
                    self.measured_ale_range.append(-irf_localization_error[0])
                    self.measured_ale_azimuth.append(-irf_localization_error[1])
                else:
                    log.debug("Target not valid, results discarded")

        self.n_views = len(self.swath)

        return True
