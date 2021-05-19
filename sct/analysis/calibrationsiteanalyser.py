# -*- coding: utf-8 -*-

"""
Calibration site analysis module
"""

import logging

log = logging.getLogger(__name__)

import numpy as np

import arepytools.constants as cst
from arepytools.geometry.conversions import xyz2llh, llh2xyz

from sct.analysis.setanalyser import SETAnalyser
from sct.analysis.platetectonicsanalyser import PlateTectonicsAnalyser
from sct.analysis.calibrationtargetanalyser import CalibrationTargetAnalyser
from sct.analysis.troposphericdelayanalyser import TroposphericDelayAnalyser
from sct.analysis.ionosphericdelayanalyser import IonosphericDelayAnalyser


class CalibrationSiteAnalyser:
    """CalibrationSiteAnalyser class"""

    def __init__(self, sar_product, calibration_targets):
        """Initialise CalibrationSiteAnalyser object

        :param sar_product: SAR product object (type depends on the mission)
        :param calibration_targets: Calibration targets as list of CalibrationTarget objects
        :type calibration_targets: list
        """

        log.info("Initialize calibration site analyser")

        self.sar_product = sar_product
        self.calibration_targets = calibration_targets
        self.calibration_targets_analysis_results = []

    def __compute_satellite_xyz_coordinates(
        self,
    ):
        """Compute calibration targets XYZ coordinates

        :return: Tuple: Satellite XYZ coordinates at which calibration targets are seen, calibration targets XYZ coordinates, both as numpy arrays of size 3xN
        :rtype: numpy.ndarray, numpy.ndarray
        """

        fc_hz = self.sar_product.dataset_info[0].fc_hz
        calibration_target_xyz = np.zeros((3, len(self.calibration_targets)))
        sat_xyz_coordinates = np.zeros((3, len(self.calibration_targets)))
        for ct in range(calibration_target_xyz.shape[1]):
            calibration_target_xyz[:, ct] = self.calibration_targets[ct].xyz.flatten()
            t_az, _ = self.sar_product.general_sar_orbit[0].earth2sat(
                calibration_target_xyz[:, ct], 0.0, cst.LIGHT_SPEED / fc_hz
            )
            sat_xyz_coordinates[:, ct] = self.sar_product.general_sar_orbit[0].get_position(t_az).flatten()

        return sat_xyz_coordinates, calibration_target_xyz

    def apply_plate_tectonics_corrections(
        self,
    ):
        """Update calibration targets XYZ coordinates applying plate tectonics corrections

        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Apply plate tectonics correction")

        # Loop over calibration targets
        for calibration_target in self.calibration_targets:

            log.debug(
                "Calibration target: {} (plate: {}, survey time: {})".format(
                    calibration_target.id, calibration_target.plate, calibration_target.survey_time
                )
            )

            # Get plate tectonics correction
            pta = PlateTectonicsAnalyser(
                calibration_target.xyz, calibration_target.plate, calibration_target.survey_time
            )
            xyz_updated = pta.get_updated_coordinates(self.sar_product.mid_time)

            # Update target coordinates
            calibration_target.update_coordinates(xyz_updated)

        return True

    def apply_set_corrections(
        self,
    ):
        """Update calibration targets XYZ coordinates applying Solid Earth Tides (SET) corrections

        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Apply SET correction")

        # Loop over calibration targets
        for calibration_target in self.calibration_targets:

            log.debug("Calibration target: {}".format(calibration_target.id))

            # Get SET correction
            sa = SETAnalyser(calibration_target.xyz)
            xyz_updated = sa.get_updated_coordinates(self.sar_product.mid_time)

            # Update target coordinates
            calibration_target.update_coordinates(xyz_updated)

        return True

    def analyse_calibration_targets(self, irf_analysis_conf):
        """Analyse calibration targets

        :param irf_analysis_conf: Calibration target analysis configuration
        :type irf_analysis_conf: dict
        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Analyse calibration targets")

        # Loop over calibration targets
        for calibration_target in self.calibration_targets:

            log.debug("Calibration target: {}".format(calibration_target.id))

            # Compute IRF parameters
            cta = CalibrationTargetAnalyser(self.sar_product, calibration_target)
            cta.analyse_calibration_target(irf_analysis_conf["maximum_ale"])

            # Store results
            for n in range(cta.n_views):
                results = {}
                results["Product name"] = cta.sar_product.name
                results["Target ID"] = cta.calibration_target.id
                results["Swath"] = cta.swath[n]
                results["Burst"] = cta.burst[n]
                results["Polarization"] = cta.polarization[n]
                results["Orbit direction"] = cta.sar_product.orbit_direction
                results["Orbit type"] = cta.sar_product.orbit_type
                results["RCS [dB]"] = cta.rcs[n]
                results["Clutter [dB]"] = cta.clutter[n]
                results["SCR [dB]"] = cta.scr[n]
                results["Peak range position []"] = cta.position_range[n]
                results["Peak azimuth position []"] = cta.position_azimuth[n]
                results["Incidence angle [deg]"] = cta.incidence_angle[n]
                results["Look angle [deg]"] = cta.look_angle[n]
                results["Squint angle [deg]"] = cta.squint_angle[n]
                results["Range resolution [m]"] = cta.resolution_range[n]
                results["Azimuth resolution [m]"] = cta.resolution_azimuth[n]
                results["Measured range ALE [m]"] = cta.measured_ale_range[n]
                results["Measured azimuth ALE [m]"] = cta.measured_ale_azimuth[n]

                self.calibration_targets_analysis_results.append(results)

        return True

    def compute_range_corrections(
        self,
    ):
        """Compute mission-dependent ALE corrections along range direction

        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Compute range corrections")

        # Loop over calibration targets and views
        for results_ind, results in enumerate(self.calibration_targets_analysis_results):

            log.debug(
                "Calibration target: {} (swath: {}, burst {})".format(
                    results["Target ID"], results["Swath"], results["Burst"]
                )
            )

            # Get range correction
            calibration_target_xyz = [
                calibration_target
                for calibration_target in self.calibration_targets
                if calibration_target.id == results["Target ID"]
            ][0].xyz
            doppler_shift_correction = self.sar_product.get_range_correction(
                results["Swath"],
                results["Burst"],
                results["Peak range position []"],
                results["Peak azimuth position []"],
                calibration_target_xyz,
            )

            # Store results
            self.calibration_targets_analysis_results[results_ind]["Doppler shift [m]"] = doppler_shift_correction

        return True

    def compute_azimuth_corrections(
        self,
    ):
        """Compute mission-dependent ALE corrections along azimuth direction

        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Compute azimuth corrections")

        # Loop over calibration targets and views
        for results_ind, results in enumerate(self.calibration_targets_analysis_results):

            log.debug(
                "Calibration target: {} (swath: {}, burst: {})".format(
                    results["Target ID"], results["Swath"], results["Burst"]
                )
            )

            # Get azimuth correction
            calibration_target_xyz = [
                calibration_target
                for calibration_target in self.calibration_targets
                if calibration_target.id == results["Target ID"]
            ][0].xyz
            (
                bistatic_delay_correction,
                instrument_timing_correction,
                fmrate_shift_correction,
            ) = self.sar_product.get_azimuth_correction(
                results["Swath"],
                results["Burst"],
                results["Peak range position []"],
                results["Peak azimuth position []"],
                calibration_target_xyz,
            )

            # Store results
            self.calibration_targets_analysis_results[results_ind]["Bistatic delay [m]"] = bistatic_delay_correction
            self.calibration_targets_analysis_results[results_ind][
                "Instrument timing [m]"
            ] = instrument_timing_correction
            self.calibration_targets_analysis_results[results_ind]["FM rate shift [m]"] = fmrate_shift_correction

        return True

    def compute_ionospheric_delay_corrections(self, ionospheric_delay_analysis_conf):
        """Compute ALE corrections due to ionospheric propagation delay

        :param ionospheric_delay_analysis_conf: Ionospheric delay corrections computation configuration
        :type ionospheric_delay_analysis_conf: dict
        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Compute ionospheric delay correction")

        # Get ionospheric delay correction
        sat_xyz_coordinates, calibration_target_xyz = self.__compute_satellite_xyz_coordinates()
        fc_hz = self.sar_product.dataset_info[0].fc_hz

        ida = IonosphericDelayAnalyser(
            calibration_target_xyz,
            ionospheric_delay_analysis_conf["ionosphere_maps_dir"],
            ionospheric_delay_analysis_conf["ionosphere_analysis_center"],
        )

        ionospheric_delay = ida.get_ionospheric_delay(
            sat_xyz_coordinates,
            self.sar_product.mid_time,
            fc_hz,
            ionospheric_delay_analysis_conf["ionospheric_delay_scaling_factor"],
        )

        # Store results
        for results_ind, results in enumerate(self.calibration_targets_analysis_results):
            ct_ind = [ct_ind for ct_ind, ct in enumerate(self.calibration_targets) if ct.id == results["Target ID"]][0]
            self.calibration_targets_analysis_results[results_ind]["Ionospheric delay [m]"] = -float(
                ionospheric_delay[ct_ind]
            )

        return True

    def compute_tropospheric_delay_corrections(self, tropospheric_delay_analysis_conf):
        """Compute ALE corrections due to tropospheric propagation delay

        :param tropospheric_delay_analysis_conf: Tropospheric delay corrections computation configuration
        :type tropospheric_delay_analysis_conf: dict
        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Compute tropospheric delay correction")

        # Get tropospheric delay correction
        sat_xyz_coordinates, calibration_target_xyz = self.__compute_satellite_xyz_coordinates()

        tda = TroposphericDelayAnalyser(
            calibration_target_xyz,
            tropospheric_delay_analysis_conf["troposphere_maps_dir"],
            tropospheric_delay_analysis_conf["troposphere_maps_model"],
            tropospheric_delay_analysis_conf["troposphere_maps_resolution"],
            tropospheric_delay_analysis_conf["troposphere_maps_version"],
        )

        tropospheric_delay_hydrostatic, tropospheric_delay_wet = tda.get_tropospheric_delay(
            sat_xyz_coordinates, self.sar_product.mid_time
        )

        # Store results
        for results_ind, results in enumerate(self.calibration_targets_analysis_results):
            ct_ind = [ct_ind for ct_ind, ct in enumerate(self.calibration_targets) if ct.id == results["Target ID"]][0]
            self.calibration_targets_analysis_results[results_ind]["Tropospheric delay (h) [m]"] = -float(
                tropospheric_delay_hydrostatic[ct_ind]
            )
            self.calibration_targets_analysis_results[results_ind]["Tropospheric delay (w) [m]"] = -float(
                tropospheric_delay_wet[ct_ind]
            )

        return True
