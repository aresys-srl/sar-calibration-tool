# -*- coding: utf-8 -*-

"""
Calibration site module
"""

import logging

log = logging.getLogger(__name__)

import copy
import io
import json
import numpy as np
import os
import pandas as pd
from shapely.geometry import Polygon

from arepytools.timing.precisedatetime import PreciseDateTime

from sct.analysis.calibrationsiteanalyser import CalibrationSiteAnalyser
from sct.analysis.calibrationtarget import CalibrationTarget
from sct.sarproduct.sarproduct import get_sar_product_mission
from sct.sarproduct.sentinel1product import Sentinel1Product


class CalibrationSite:
    """CalibrationSite class"""

    def __init__(self, id, calibration_targets_file, bounding_polygon_file):
        """Initialise CalibrationSite object

        :param id: Calibration site identifier
        :type id: str
        :param calibration_targets_file: Path to calibration targets file
        :type calibration_targets_file: str
        :param bounding_polygon_file: Path to bounding polygon file
        :type bounding_polygon_file: str
        """

        log.info("Initialize calibration site")
        log.info("  Calibration site: {}".format(id))

        self.id = id
        self.calibration_targets, self.bounding_polygon = self.__load_calibration_site(
            calibration_targets_file, bounding_polygon_file
        )

        self.overpasses = []
        self.calibration_db = None

    def __str__(self):
        """CalibrationSite class __str__ function

        :return: Calibration site description and main parameters
        :rtype: str
        """

        buffer = io.StringIO()
        self.calibration_db.info(buf=buffer)

        s = (
            "Calibration Site: {} (id: {})\n".format(self.id.replace("_", " ").title(), self.id)
            + "Lat\\Lon Area: {}:{}deg \\ {}:{}deg\n".format(
                min(self.bounding_polygon[:, 0]),
                max(self.bounding_polygon[:, 0]),
                min(self.bounding_polygon[:, 1]),
                max(self.bounding_polygon[:, 1]),
            )
            + "Number of overpasses: {}\n".format(len(self.overpasses))
            + "Database: \n{}".format(buffer.getvalue())
        )

        return s

    def __load_calibration_site(self, calibration_targets_file, bounding_polygon_file):
        """Load calibration site

        :param calibration_targets_file: Path to calibration targets file
        :type calibration_targets_file: str
        :param bounding_polygon_file: Path to bounding polygon file
        :type bounding_polygon_file: str
        :return: Tuple: Calibration targets as list of CalibrationTarget objects, bounding polygon as numpy array of size Nx2
        :rtype: list, numpy.ndarray
        """

        # Read bounding polygon file
        log.info("Read bounding polygon file")

        if not os.path.isfile(bounding_polygon_file):
            raise FileNotFoundError("Bounding polygon file not found. Calibration site can't be initialised.")

        with open(bounding_polygon_file) as f:
            bounding_polygon = json.load(f)
        bounding_polygon = np.asarray(bounding_polygon["features"][0]["geometry"]["coordinates"][0])
        bounding_polygon = bounding_polygon[:, [1, 0]]

        # Read calibration targets file
        log.info("Read calibration targets file")

        if not os.path.isfile(calibration_targets_file):
            raise FileNotFoundError("Calibration targets file not found. Calibration site can't be initialised.")

        df = pd.read_excel(calibration_targets_file)
        calibration_targets = []
        for t in range(len(df)):
            df_curr = df.iloc[t]
            calibration_target = CalibrationTarget(
                df_curr["ID"],
                df_curr["Type"],
                df_curr["Latitude [deg]"],
                df_curr["Longitude [deg]"],
                df_curr["Altitude [m]"],
                {
                    "H/H": df_curr["RCS_HH [dB]"],
                    "H/V": df_curr["RCS_HV [dB]"],
                    "V/H": df_curr["RCS_VH [dB]"],
                    "V/V": df_curr["RCS_VV [dB]"],
                },
                df_curr["Delay [s]"],
                df_curr["Plate"],
                PreciseDateTime().set_from_utc_string(df_curr["Survey Time [UTC]"]),
                PreciseDateTime().set_from_utc_string(df_curr["Validity Start [UTC]"]),
                PreciseDateTime().set_from_utc_string(df_curr["Validity Stop [UTC]"]),
            )
            calibration_targets.append(calibration_target)

        return calibration_targets, bounding_polygon

    def init_calibration_db(self, calibration_db_file):
        """Initialise calibration database

        :param calibration_db_file: Path to calibration database file
        :type calibration_db_file: str
        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Initialize calibration database")

        # Initialize calibration database as pandas dataframe
        column_names = [
            "Product name",
            "Target ID",
            "Swath",
            "Burst",
            "Polarization",
            "Orbit direction",
            "Orbit type",
            "RCS [dB]",
            "Clutter [dB]",
            "SCR [dB]",
            "Peak range position []",
            "Peak azimuth position []",
            "Incidence angle [deg]",
            "Look angle [deg]",
            "Squint angle [deg]",
            "Range resolution [m]",
            "Azimuth resolution [m]",
            "Measured range ALE [m]",
            "Measured azimuth ALE [m]",
            "Bistatic delay [m]",
            "Instrument timing [m]",
            "Doppler shift [m]",
            "FM rate shift [m]",
            "Ionospheric delay [m]",
            "Tropospheric delay (h) [m]",
            "Tropospheric delay (w) [m]",
        ]
        calibration_db = pd.DataFrame(columns=column_names)

        # Save calibration database file
        if os.path.isfile(calibration_db_file):
            raise FileExistsError("Calibration database file already exists and can't be overwritten.")

        calibration_db.to_excel(calibration_db_file, index=False)

        return True

    def load_calibration_db(self, calibration_db_file):
        """Load calibration database

        :param calibration_db_file: Path to calibration database file
        :type calibration_db_file: str
        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Load calibration database")

        # Import calibration database file as pandas dataframe
        if not os.path.isfile(calibration_db_file):
            raise FileNotFoundError("Calibration database file not found. Calibration site can't be initialised.")

        self.calibration_db = pd.read_excel(calibration_db_file)

        # Update overpasses list
        # NOTE The following commands keep unique elements in the list of SAR products and preserve their order
        seen = set()
        self.overpasses = [x for x in self.calibration_db["Product name"] if not (x in seen or seen.add(x))]

        return True

    def __is_already_analysed(self, sar_product_dir):
        """Check if SAR product has already been analysed and added to the calibration database

        :param sar_product_dir: Path to SAR product folder
        :type sar_product_dir: str
        :return: Status (True if yes, False otherwise)
        :rtype: bool
        """

        sar_product_name = os.path.basename(sar_product_dir)

        return sar_product_name in self.overpasses

    def __is_overlapping(self, sar_product_footprint):
        """Check if SAR product footprint is overlapping with calibration site bounding polygon

        :param sar_product_footprint: SAR product footprint as numpy array of size Nx2
        :type sar_product_footprint: numpy.ndarray
        :return: Status (True if yes, False otherwise)
        :rtype: bool
        """

        p1 = Polygon(self.bounding_polygon)
        p2 = Polygon(sar_product_footprint)

        return p1.intersects(p2)

    def get_overpass_results(self, sar_product_name):
        """If it has already been analysed, get results relative to provided SAR product from calibration database

        :param sar_product_name: SAR product name
        :type sar_product_name: str
        :return: SAR product results
        :rtype: pandas.core.frame.DataFrame
        """

        if self.__is_already_analysed(sar_product_name):
            return self.calibration_db[self.calibration_db["Product name"] == sar_product_name]
        else:
            log.info("Provided SAR product not present in database.")
            return None

    def add_overpass(self, sar_product_dir, orbit_file=None):
        """Add SAR product to calibration site overpasses. If SAR product has already been analysed returns correspoding results,
        otherwise initialise it and returns the corresponding object

        :param sar_product_dir: Path to SAR product folder
        :type sar_product_dir: str
        :param orbit_file: Path to orbit file, defaults to None
        :type orbit_file: str, optional
        :return: SAR product object (type depends on the mission)
        """

        log.info("Add overpass")
        log.info("  SAR product: {}".format(sar_product_dir))
        log.info("  Orbit file: {}".format(orbit_file))

        # Check if calibration database has been loaded
        if self.calibration_db is None:
            raise RuntimeError("Calibration database not loaded.")

        # Check if overpass (SAR product) is already present in the list and in case return corresponding results
        if self.__is_already_analysed(sar_product_dir):
            log.info("Provided SAR product already present in database. Corresponding results returned.")
            return self.get_overpass_results(sar_product_dir)

        # If not, initialize SAR product
        sar_product_mission = get_sar_product_mission(sar_product_dir)
        if sar_product_mission == "SENTINEL-1":
            sp = Sentinel1Product(sar_product_dir, orbit_file)
        elif sar_product_mission == "BIOMASS":
            raise NotImplementedError("BIOMASS mission data not yet supported.")
        else:
            raise RuntimeError("{} mission not supported.".format(sar_product_mission))

        # Check overlap between SAR product footprint and calibration site bounding polygon
        if self.__is_overlapping(sp.footprint) == True:
            # If overlapping, add SAR product to calibration site overpasses and return corresponding object
            log.info("Provided SAR product overlapping with current calibration site bounding polygon.")
            self.overpasses.append(sp.name)
            return sp
        else:
            # Otherwise, return error
            raise RuntimeError("Provided SAR product not overlapping with current calibration site bounding polygon.")

    def __update_calibration_db(self, calibration_targets_analysis_results):
        """Update calibration database adding SAR product results

        :param calibration_targets_analysis_results: Results of SAR product analysis as a list of dictionaries, one for each calibration target
        :type calibration_targets_analysis_results: list of dict
        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Update calibration database")

        self.calibration_db = self.calibration_db.append(pd.DataFrame(calibration_targets_analysis_results))

        return True

    def analyse_overpass(self, sar_product, calibration_site_analysis_conf):
        """Analyse SAR product

        :param sar_product: SAR product object (type depends on the mission)
        :param calibration_site_analysis_conf: Calibration site analysis configuration
        :type calibration_site_analysis_conf: dict
        :return: SAR product results
        :rtype: pandas.core.frame.DataFrame
        """

        log.info("Analyse overpass")

        # Read configuration parameters
        processing_steps_conf = calibration_site_analysis_conf["processing_steps"]
        irf_analysis_conf = calibration_site_analysis_conf["irf_analysis"]
        ionospheric_delay_analysis_conf = calibration_site_analysis_conf["ionospheric_delay_analysis"]
        tropospheric_delay_analysis_conf = calibration_site_analysis_conf["tropospheric_delay_analysis"]

        # Select only calibration targets visible by SAR product and valid
        log.info("Select only calibration targets visible by SAR product and valid")
        calibration_targets = []
        for calibration_target in self.calibration_targets:
            if calibration_target.is_visible(sar_product.footprint) and calibration_target.is_valid(
                sar_product.start_time
            ):
                log.debug("Calibration target {} visible and valid.".format(calibration_target.id))
                calibration_targets.append(
                    copy.deepcopy(calibration_target)
                )  # NOTE Force copy to avoid modifying original calibration targets coordinates
            else:
                log.debug("Calibration target {} not visible or valid.".format(calibration_target.id))

        # Initialize calibration site analyser
        csa = CalibrationSiteAnalyser(sar_product, calibration_targets)

        # Apply plate tectonics correction
        if processing_steps_conf["apply_plate_tectonics_correction"]:
            csa.apply_plate_tectonics_corrections()

        # Apply SET correction
        if processing_steps_conf["apply_set_correction"]:
            csa.apply_set_corrections()

        # Analyse calibration targets
        csa.analyse_calibration_targets(irf_analysis_conf)

        # Compute range corrections
        if processing_steps_conf["apply_range_corrections"]:
            csa.compute_range_corrections()

        # Compute azimuth corrections
        if processing_steps_conf["apply_azimuth_corrections"]:
            csa.compute_azimuth_corrections()

        # Compute ionospheric delay correction
        if processing_steps_conf["apply_ionospheric_delay_correction"]:
            csa.compute_ionospheric_delay_corrections(ionospheric_delay_analysis_conf)

        # Compute tropospheric delay correction
        if processing_steps_conf["apply_tropospheric_delay_correction"]:
            csa.compute_tropospheric_delay_corrections(tropospheric_delay_analysis_conf)

        # Store results
        self.__update_calibration_db(csa.calibration_targets_analysis_results)

        return self.get_overpass_results(sar_product.name)

    def save_calibration_db(self, calibration_db_file, force_overwrite=False):
        """Save calibration database

        :param calibration_db_file: Path to calibration database file
        :type calibration_db_file: str
        :param force_overwrite: Force overwrite flag, defaults to False
        :type force_overwrite: bool, optional
        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.info("Save calibration database")

        if os.path.isfile(calibration_db_file):
            if not force_overwrite:
                raise FileExistsError(
                    "Provided file already exists and can't be overwritten. Use force_overwrite parameter."
                )
            else:
                log.info("Provided file already exists and will be overwritten.")

        self.calibration_db.to_excel(calibration_db_file, index=False)

        return True
