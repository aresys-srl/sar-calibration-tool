# -*- coding: utf-8 -*-

"""
Configuration module
"""

import logging

log = logging.getLogger(__name__)

from distutils.util import strtobool
import xml.etree.ElementTree as ET


class Configuration:
    """Configuration class"""

    def __init__(self, configuration_file=None):
        """Initialise Configuration object

        :param configuration_file: Path to configuration file, defaults to None
        :type configuration_file: str, optional
        """

        log.info("Initialize configuration")

        if configuration_file is not None:
            # Read configuration file
            self.calibration_site_analysis_conf = self.__read_configuration_file(configuration_file)

            # Check configuration parameters
            if not self.__check_configuration_parameters():
                raise RuntimeError("Invalid configuration parameters provided.")

        else:
            # Use default parameters
            self.calibration_site_analysis_conf = self.__get_default_configuration_parameters()

    def __read_configuration_file(self, configuration_file):
        """Read configuration file

        :param configuration_file: Path to configuration file
        :type configuration_file: str
        :return: Calibration site analysis configuration
        :rtype: dict
        """

        log.info("Read configuration parameters from file")

        # Read configuration file
        tree = ET.parse(configuration_file)
        root = tree.getroot()

        calibration_site_analysis = root.find("calibration_site_analysis")

        element = calibration_site_analysis.find("processing_steps")
        processing_steps = {}
        processing_steps["apply_plate_tectonics_correction"] = bool(
            strtobool(element.find("apply_plate_tectonics_correction").text)
        )
        processing_steps["apply_set_correction"] = bool(strtobool(element.find("apply_set_correction").text))
        processing_steps["apply_range_corrections"] = bool(strtobool(element.find("apply_range_corrections").text))
        processing_steps["apply_azimuth_corrections"] = bool(strtobool(element.find("apply_azimuth_corrections").text))
        processing_steps["apply_ionospheric_delay_correction"] = bool(
            strtobool(element.find("apply_ionospheric_delay_correction").text)
        )
        processing_steps["apply_tropospheric_delay_correction"] = bool(
            strtobool(element.find("apply_tropospheric_delay_correction").text)
        )

        element = calibration_site_analysis.find("irf_analysis")
        irf_analysis = {}
        maximum_ale = element.find("maximum_ale").text
        irf_analysis["maximum_ale"] = None if maximum_ale == "None" else float(maximum_ale)

        element = calibration_site_analysis.find("ionospheric_delay_analysis")
        ionospheric_delay_analysis = {}
        ionospheric_delay_analysis["ionosphere_maps_dir"] = element.find("ionosphere_maps_dir").text
        ionospheric_delay_analysis["ionosphere_analysis_center"] = element.find("ionosphere_analysis_center").text
        ionospheric_delay_analysis["ionospheric_delay_scaling_factor"] = float(
            element.find("ionospheric_delay_scaling_factor").text
        )

        element = calibration_site_analysis.find("tropospheric_delay_analysis")
        tropospheric_delay_analysis = {}
        tropospheric_delay_analysis["troposphere_maps_dir"] = element.find("troposphere_maps_dir").text
        tropospheric_delay_analysis["troposphere_maps_model"] = element.find("troposphere_maps_model").text
        tropospheric_delay_analysis["troposphere_maps_resolution"] = element.find("troposphere_maps_resolution").text
        tropospheric_delay_analysis["troposphere_maps_version"] = element.find("troposphere_maps_version").text

        calibration_site_analysis = {
            "processing_steps": processing_steps,
            "irf_analysis": irf_analysis,
            "ionospheric_delay_analysis": ionospheric_delay_analysis,
            "tropospheric_delay_analysis": tropospheric_delay_analysis,
        }

        return calibration_site_analysis

    def __get_default_configuration_parameters(
        self,
    ):
        """Get default configuration parameters

        :return: Calibration site analysis configuration
        :rtype: dict
        """

        log.info("Use default configuration parameters")

        # Use default configuration parameters
        processing_steps = {
            "apply_plate_tectonics_correction": True,
            "apply_set_correction": True,
            "apply_range_corrections": True,
            "apply_azimuth_corrections": True,
            "apply_ionospheric_delay_correction": True,
            "apply_tropospheric_delay_correction": True,
        }
        irf_analysis = {
            "maximum_ale": None,
        }
        ionospheric_delay_analysis = {
            "ionosphere_maps_dir": "./data/auxiliary_files/ionosphere_maps",
            "ionosphere_analysis_center": "cod",
            "ionospheric_delay_scaling_factor": 1.0,
        }
        tropospheric_delay_analysis = {
            "troposphere_maps_dir": "./data/auxiliary_files/troposphere_maps",
            "troposphere_maps_model": "GRID",
            "troposphere_maps_resolution": "1x1",
            "troposphere_maps_version": "OP",
        }
        calibration_site_analysis = {
            "processing_steps": processing_steps,
            "irf_analysis": irf_analysis,
            "ionospheric_delay_analysis": ionospheric_delay_analysis,
            "tropospheric_delay_analysis": tropospheric_delay_analysis,
        }

        return calibration_site_analysis

    def __check_configuration_parameters(
        self,
    ):
        """Check configuration parameters

        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.debug("Check configuration parameters")

        return True  # TODO
