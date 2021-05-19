# -*- coding: utf-8 -*-

"""
Sentinel-1 SAR product module
"""

import logging

log = logging.getLogger(__name__)

from glob import glob
import numpy as np
import os
import coda
import rasterio
from rasterio.windows import Window
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import arepytools.constants as cst

# import arepytools.geometry.generalsarorbit as generalsarorbit   # TODO ..with this
import arepytools.io.metadata as metadata
from arepytools.math import genericpoly
from arepytools.timing.precisedatetime import PreciseDateTime

from sct.sarproduct.sarproduct import SARProduct, EDataQuantity
from sct.orbit.eeorbit import EEOrbit, EOrbitType
import sct.support.generalsarorbit_extended as generalsarorbit  # TODO Replace this..


class Sentinel1Product(SARProduct):
    """Sentinel1Product class"""

    orbit_type_dict = {
        "AUX_PRE": EOrbitType.predicted.value,
        "AUX_RES": EOrbitType.restituted.value,
        "AUX_POE": EOrbitType.precise.value,
    }

    tx_pulse_latch_time = 1.439e-6  # [s]

    def __init__(self, sar_product_dir, orbit_file=None):
        """Initialise Sentinel1Product object

        :param sar_product_dir: Path to SAR product folder
        :type sar_product_dir: str
        :param orbit_file: Path to orbit file, defaults to None
        :type orbit_file: str, optional
        """

        log.debug("Initialize SAR product")

        SARProduct.__init__(self, sar_product_dir, orbit_file)

        if self.orbit_file is not None:
            self.__use_external_orbit = True
        else:
            self.__use_external_orbit = False

        self.__channels = None
        self.__measurement_files = None
        self.__annotation_files = None
        self.__calibration_files = None
        self.__noise_files = None
        self.__manifest_file = None

        self.__init_product_paths()

        self.__scaling_factors = None

        self.__read_metadata()

        self._set_product_roi()

    def __init_product_paths(
        self,
    ):
        """Initialise SAR product internal paths

        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        # Initialize paths of SAR product single files
        # - Number of channels
        measurement_files = glob(os.path.join(self.sar_product_dir, "measurement", "*.tiff"))
        self.__channels = len(measurement_files)
        self.__measurement_files = [""] * self.__channels
        self.__annotation_files = [""] * self.__channels
        self.__calibration_files = [""] * self.__channels
        self.__noise_files = [""] * self.__channels
        self.__manifest_file = ""

        # - Measurement files
        for f in measurement_files:
            measurement_file = os.path.basename(f)
            self.__measurement_files[int(measurement_file[61:64]) - 1] = f

        # - Annotation files
        annotation_files = glob(os.path.join(self.sar_product_dir, "annotation", "*.xml"))
        for f in annotation_files:
            annotation_file = os.path.basename(f)
            self.__annotation_files[int(annotation_file[61:64]) - 1] = f

        # - Calibration files
        calibration_files = glob(os.path.join(self.sar_product_dir, "annotation", "calibration", "calibration-*.xml"))
        for f in calibration_files:
            calibration_file = os.path.basename(f)
            self.__calibration_files[int(calibration_file[73:76]) - 1] = f

        # - Noise files
        noise_files = glob(os.path.join(self.sar_product_dir, "annotation", "calibration", "noise-*.xml"))
        for f in noise_files:
            noise_file = os.path.basename(f)
            self.__noise_files[int(noise_file[67:70]) - 1] = f

        # - Manifest file
        self.__manifest_file = os.path.join(self.sar_product_dir, "manifest.safe")

        return True

    def read_data(self, channel, roi=None, n_rg=None, n_az=None):
        """Read SAR product data portion

        :param channel: SAR product channel
        :type channel: int
        :param roi: Data portion coordinates as numpy array of size 4 (first azimuth, first range, number of lines, number of samples), defaults to None
        :type roi: numpy.ndarray, optional
        :param n_rg: Channel number of samples, defaults to None
        :type n_rg: int, optional
        :param n_az: Channel number of lines, defaults to None
        :type n_az: int, optional
        :return: Data portion as numpy array of size Nsamples x Nlines
        :rtype: numpy.ndarray
        """

        # log.debug('Read SAR product data portion')

        # Open measurement file
        with rasterio.open(self.__measurement_files[channel], mode="r") as file_object:

            # If roi is not provided, read the whole data
            if roi is None:
                roi = [0, 0, file_object.height, file_object.width]  # [0, 0, lines, samples]

            # If data dimensions are provided, verify that input roi doesn't exceed them and, in case, limit it
            roi_mod = roi.copy()
            if roi_mod[0] < 0:
                roi_padleft = np.abs(roi_mod[0])
                roi_mod[0] = 0
                roi_mod[2] = roi_mod[2] - roi_padleft
            else:
                roi_padleft = 0

            if roi_mod[1] < 0:
                roi_padup = np.abs(roi_mod[1])
                roi_mod[1] = 0
                roi_mod[3] = roi_mod[3] - roi_padup
            else:
                roi_padup = 0

            roi_padright = 0
            roi_paddown = 0
            if (n_rg is not None) and (n_az is not None):
                if roi_mod[0] + roi_mod[2] > n_az:
                    roi_padright = roi_mod[0] + roi_mod[2] - n_az
                    roi_mod[2] = roi_mod[2] - roi_padright

                if roi_mod[1] + roi_mod[3] > n_rg:
                    roi_paddown = roi_mod[1] + roi_mod[3] - n_rg
                    roi_mod[3] = roi_mod[3] - roi_paddown

            # Read data roi
            # TODO Move parts before and after this section to sarproduct.py
            file_window = Window(col_off=roi_mod[1], row_off=roi_mod[0], width=roi_mod[3], height=roi_mod[2])
            data = file_object.read(indexes=1, window=file_window)
            data = data.transpose()  # TODO Adopt Python convention, i.e. x=range y=azimuth

            # In case input roi has been limited, pad accordingly data read
            if roi_padleft > 0:
                pad = np.zeros((roi_mod[3], roi_padleft))
                data = np.concatenate((pad, data), axis=1)

            if roi_padup > 0:
                pad = np.zeros((roi_padup, roi_mod[2] + roi_padleft))
                data = np.concatenate((pad, data), axis=0)

            if (n_rg is not None) and (n_az is not None):
                if roi_padright > 0:
                    pad = np.zeros((roi_mod[3] + roi_padup, roi_padright))
                    data = np.concatenate((data, pad), axis=1)

                if roi_paddown > 0:
                    pad = np.zeros((roi_paddown, roi_mod[2] + roi_padleft + roi_padright))
                    data = np.concatenate((data, pad), axis=0)

            # Properly scale data
            data = data * self.__scaling_factors[channel]

        return data

    def __read_metadata(
        self,
    ):
        """Read SAR product metadata

        :return: Status (True for success, False for unsuccess)
        :rtype: bool
        """

        log.debug("Read SAR product metadata")

        # Read SAR product metadata
        # - Read SAR product name fields
        log.debug("  Name fields")

        name = os.path.basename(self.sar_product_dir)

        self.name = name
        self.mission = name[0:3]
        self.acquisition_mode = name[4:6]
        self.type = name[7:10]
        self.polarization = name[14:16]
        self.start_time = PreciseDateTime().set_from_isoformat(name[17:32])
        self.stop_time = PreciseDateTime().set_from_isoformat(name[33:48])
        self.mid_time = self.start_time + (self.stop_time - self.start_time) / 2
        self.orbit_number = int(name[49:55])

        # - Read annotation files
        for channel in range(self.__channels):

            log.debug("  Annotation file (channel {})".format(channel))

            # Open annotation file
            file_handle = coda.open(self.__annotation_files[channel])

            # Fetch content
            file_data = coda.fetch(file_handle)

            # Set RasterInfo list
            samples = int(file_data.product.imageAnnotation.imageInformation.numberOfSamples)
            samples_start = float(file_data.product.imageAnnotation.imageInformation.slantRangeTime)
            samples_start_unit = "s"
            samples_step = 1 / float(file_data.product.generalAnnotation.productInformation.rangeSamplingRate)
            samples_step_unit = samples_start_unit
            lines = int(file_data.product.imageAnnotation.imageInformation.numberOfLines)
            lines_start = PreciseDateTime().set_from_utc_string(
                file_data.product.imageAnnotation.imageInformation.productFirstLineUtcTime
            )
            lines_start_unit = "Mjd"
            lines_step = float(file_data.product.imageAnnotation.imageInformation.azimuthTimeInterval)
            lines_step_unit = "s"
            metadata_ri = metadata.RasterInfo(
                lines,
                samples,
                celltype="FLOAT_COMPLEX",
                filename=None,
                header_offset_bytes=0,
                row_prefix_bytes=0,
                byteorder="LITTLEENDIAN",
            )
            metadata_ri.set_lines_axis(lines_start, lines_start_unit, lines_step, lines_step_unit)
            metadata_ri.set_samples_axis(samples_start, samples_start_unit, samples_step, samples_step_unit)

            self.raster_info_list.append(metadata_ri)

            # Set BurstInfo list
            burst_count = file_data.product.swathTiming.burstList.burst.size
            range_start_time = float(file_data.product.imageAnnotation.imageInformation.slantRangeTime)
            lines = int(file_data.product.swathTiming.linesPerBurst)
            metadata_bi = metadata.BurstInfo()
            for b in range(burst_count):
                azimuth_start_time = PreciseDateTime().set_from_utc_string(
                    file_data.product.swathTiming.burstList.burst[b].azimuthTime
                )
                metadata_bi.add_burst(range_start_time, azimuth_start_time, lines, burst_center_azimuth_shift_i=None)

            self.burst_info_list.append(metadata_bi)

            # Set DataSetInfo
            if channel == 0:
                sensor_name = file_data.product.adsHeader.missionId
                acquisition_mode = file_data.product.adsHeader.mode
                projection = file_data.product.generalAnnotation.productInformation.projection.upper()
                fc_hz = float(file_data.product.generalAnnotation.productInformation.radarFrequency)
                metadata_di = metadata.DataSetInfo(acquisition_mode, fc_hz)
                metadata_di.sensor_name = sensor_name
                metadata_di.projection = projection
                metadata_di.side_looking = "RIGHT"

                self.dataset_info.append(metadata_di)

            # Set SwathInfo list
            swath = file_data.product.adsHeader.swath
            polarization = file_data.product.adsHeader.polarisation
            polarization = polarization[0] + "/" + polarization[1]
            rank = int(
                file_data.product.generalAnnotation.downlinkInformationList.downlinkInformation.downlinkValues.rank
            )
            azimuth_steering_rate = (
                float(file_data.product.generalAnnotation.productInformation.azimuthSteeringRate) * np.pi / 180
            )
            acquisition_prf = float(file_data.product.generalAnnotation.downlinkInformationList.downlinkInformation.prf)
            metadata_si = metadata.SwathInfo(swath, polarization, acquisition_prf)
            metadata_si.azimuth_steering_rate_pol = (azimuth_steering_rate, 0, 0)

            self.swath_info_list.append(metadata_si)

            # Set SamplingConstants list
            frg_hz = float(file_data.product.generalAnnotation.productInformation.rangeSamplingRate)
            brg_hz = float(
                file_data.product.imageAnnotation.processingInformation.swathProcParamsList.swathProcParams.rangeProcessing.processingBandwidth
            )
            faz_hz = 1 / float(file_data.product.imageAnnotation.imageInformation.azimuthTimeInterval)
            baz_hz = float(
                file_data.product.imageAnnotation.processingInformation.swathProcParamsList.swathProcParams.azimuthProcessing.processingBandwidth
            )
            metadata_sc = metadata.SamplingConstants(frg_hz, brg_hz, faz_hz, baz_hz)

            self.sampling_constants_list.append(metadata_sc)

            # Set AcquisitionTimeline list
            # TODO Set missing elements with proper values
            missing_lines_number = 0
            missing_lines_azimuth_times = None
            swst_list = (
                file_data.product.generalAnnotation.downlinkInformationList.downlinkInformation.downlinkValues.swstList
            )
            if type(swst_list.swst) is np.ndarray:
                swst_changes_number = swst_list.swst.size
                swst_changes_azimuth_times = [
                    PreciseDateTime().set_from_utc_string(swst.azimuthTime) for swst in swst_list.swst
                ]
                swst_changes_values = [float(swst.value) for swst in swst_list.swst]
            else:
                swst_changes_number = 1
                swst_changes_azimuth_times = [PreciseDateTime().set_from_utc_string(swst_list.swst.azimuthTime)]
                swst_changes_values = [float(swst_list.swst.value)]
            noise_packets_number = 0
            noise_packets_azimuth_times = None
            internal_calibration_number = 0
            internal_calibration_azimuth_times = None
            swl_list = (
                file_data.product.generalAnnotation.downlinkInformationList.downlinkInformation.downlinkValues.swlList
            )
            if type(swl_list.swl) is np.ndarray:
                swl_changes_number = swl_list.swl.size
                swl_changes_azimuth_times = [
                    PreciseDateTime().set_from_utc_string(swl.azimuthTime) for swl in swl_list.swl
                ]
                swl_changes_values = [float(swl.value) for swl in swl_list.swl]
            else:
                swl_changes_number = 1
                swl_changes_azimuth_times = [PreciseDateTime().set_from_utc_string(swl_list.swl.azimuthTime)]
                swl_changes_values = [float(swl_list.swl.value)]
            metadata_at = metadata.AcquisitionTimeLine(
                missing_lines_number,
                missing_lines_azimuth_times,
                swst_changes_number,
                swst_changes_azimuth_times,
                swst_changes_values,
                noise_packets_number,
                noise_packets_azimuth_times,
                internal_calibration_number,
                internal_calibration_azimuth_times,
                swl_changes_number,
                swl_changes_azimuth_times,
                swl_changes_values,
            )

            self.acquisition_timeline_list.append(metadata_at)

            # Set StateVectors
            if channel == 0:
                if self.__use_external_orbit == True:
                    orbit = EEOrbit(self.orbit_file)
                    metadata_sv = metadata.StateVectors(
                        orbit.position_sv, orbit.velocity_sv, orbit.reference_time, orbit.delta_time
                    )

                    self.orbit_direction = orbit.get_orbit_direction(self.start_time)
                    self.orbit_type = orbit.type

                else:
                    sv_count = file_data.product.generalAnnotation.orbitList.orbit.size
                    position_sv = np.zeros((sv_count, 3))
                    velocity_sv = np.zeros((sv_count, 3))
                    for sv in range(sv_count):
                        position_sv[sv][0] = float(file_data.product.generalAnnotation.orbitList.orbit[sv].position.x)
                        position_sv[sv][1] = float(file_data.product.generalAnnotation.orbitList.orbit[sv].position.y)
                        position_sv[sv][2] = float(file_data.product.generalAnnotation.orbitList.orbit[sv].position.z)
                        velocity_sv[sv][0] = float(file_data.product.generalAnnotation.orbitList.orbit[sv].velocity.x)
                        velocity_sv[sv][1] = float(file_data.product.generalAnnotation.orbitList.orbit[sv].velocity.y)
                        velocity_sv[sv][2] = float(file_data.product.generalAnnotation.orbitList.orbit[sv].velocity.z)
                    reference_time = PreciseDateTime().set_from_utc_string(
                        file_data.product.generalAnnotation.orbitList.orbit[0].time
                    )
                    delta_time = (
                        PreciseDateTime().set_from_utc_string(
                            file_data.product.generalAnnotation.orbitList.orbit[1].time
                        )
                        - reference_time
                    )
                    metadata_sv = metadata.StateVectors(position_sv, velocity_sv, reference_time, delta_time)

                    current_ind = int((self.start_time - reference_time) / delta_time)
                    if velocity_sv[current_ind][2] > 0:
                        self.orbit_direction = metadata.EOrbitDirection.ascending.value
                    else:
                        self.orbit_direction = metadata.EOrbitDirection.descending.value

                    if file_data.product.imageAnnotation.processingInformation.orbitSource == "Extracted":
                        self.orbit_type = EOrbitType.downlink.value
                    elif file_data.product.imageAnnotation.processingInformation.orbitSource == "Auxiliary":
                        self.orbit_type = (
                            EOrbitType.unknown.value
                        )  # Read this information from manifest file (see below)

                self.general_sar_orbit.append(generalsarorbit.create_general_sar_orbit(metadata_sv))

            # Set DopplerCentroidVector list
            dcest_count = file_data.product.dopplerCentroid.dcEstimateList.dcEstimate.size
            dcest_method = "DATA"  # or 'GEOMETRY'
            metadata_dcl = []
            for dc in range(dcest_count):
                dcest = file_data.product.dopplerCentroid.dcEstimateList.dcEstimate[dc]
                ref_az = PreciseDateTime().set_from_utc_string(dcest.azimuthTime)
                ref_rg = float(dcest.t0)
                if dcest_method == "GEOMETRY":
                    coefficients = dcest.geometryDcPolynomial.split(" ")
                elif dcest_method == "DATA":
                    coefficients = dcest.dataDcPolynomial.split(" ")
                coefficients = [float(i) for i in coefficients]
                coefficients = [coefficients[0], coefficients[1], 0, 0, coefficients[2], 0, 0, 0, 0, 0, 0]
                metadata_dc = metadata.DopplerCentroid(ref_az, ref_rg, coefficients)
                metadata_dcl.append(metadata_dc)
            metadata_dcv = metadata.DopplerCentroidVector(metadata_dcl)

            self.dc_vector_list.append(metadata_dcv)

            # Set DopplerRateVector list
            fmest_count = file_data.product.generalAnnotation.azimuthFmRateList.azimuthFmRate.size
            metadata_fml = []
            for fm in range(fmest_count):
                fmest = file_data.product.generalAnnotation.azimuthFmRateList.azimuthFmRate[fm]
                ref_az = PreciseDateTime().set_from_utc_string(fmest.azimuthTime)
                ref_rg = float(fmest.t0)
                coefficients = fmest.azimuthFmRatePolynomial.split(" ")
                coefficients = [float(i) for i in coefficients]
                coefficients = [coefficients[0], coefficients[1], 0, 0, coefficients[2], 0, 0, 0, 0, 0, 0]
                metadata_fm = metadata.DopplerRate(ref_az, ref_rg, coefficients)
                metadata_fml.append(metadata_fm)
            metadata_fmv = metadata.DopplerRateVector(metadata_fml)

            self.dr_vector_list.append(metadata_fmv)

            # Set Ground2Slant list
            # TODO Review module for GRD products and properly set Ground2Slant objects list

            # Set Pulse list
            # TODO Set missing elements with proper values
            pulse_length = float(
                file_data.product.generalAnnotation.downlinkInformationList.downlinkInformation.downlinkValues.txPulseLength
            )
            pulse_bandwidth = float(
                file_data.product.imageAnnotation.processingInformation.swathProcParamsList.swathProcParams.rangeProcessing.totalBandwidth
            )
            pulse_sampling_rate = float(file_data.product.generalAnnotation.productInformation.rangeSamplingRate)
            pulse_energy = 1.0
            pulse_start_frequency = float(
                file_data.product.generalAnnotation.downlinkInformationList.downlinkInformation.downlinkValues.txPulseStartFrequency
            )
            pulse_start_phase = 0.0
            pulse_direction = metadata.EPulseDirection.up.value
            metadata_p = metadata.Pulse(
                pulse_length,
                pulse_bandwidth,
                pulse_sampling_rate,
                pulse_energy,
                pulse_start_frequency,
                pulse_start_phase,
                pulse_direction,
            )

            self.pulse_list.append(metadata_p)

            # Close annotation file
            status = coda.close(file_handle)

        # - Read calibration files
        self.__scaling_factors = []

        for channel in range(self.__channels):

            log.debug("  Calibration file (channel {})".format(channel))

            # Open calibration file
            file_handle = coda.open(self.__calibration_files[channel])

            # Fetch content
            file_data = coda.fetch(file_handle)

            # Set data quantity
            self.data_quantity = EDataQuantity.beta_nought.value

            # Set scaling factor
            # NOTE Beta Nought LUT is supposed to be equal to a constant value
            self.__scaling_factors.append(
                1 / float(file_data.calibration.calibrationVectorList.calibrationVector[0].betaNought.split(" ")[0])
            )

            # Close calibration file
            status = coda.close(file_handle)

        # - Read noise files
        # /

        # - Read manifest file
        log.debug("  Manifest file")

        # Open manifest file
        file_handle = coda.open(self.__manifest_file)

        # Fetch content
        file_data = coda.fetch(file_handle)
        metadata_object_list = [
            coda.get_attributes(file_handle, "XFDU/metadataSection/metadataObject[" + str(t) + "]").ID
            for t in range(len(file_data.XFDU.metadataSection.metadataObject))
        ]

        # Set orbit type
        if self.orbit_type == EOrbitType.unknown.value:
            ind = metadata_object_list.index("processing")
            resource_role_list = [
                coda.get_attributes(
                    file_handle,
                    "XFDU/metadataSection/metadataObject["
                    + str(ind)
                    + "]/metadataWrap/xmlData/processing/resource/processing/resource["
                    + str(t)
                    + "]",
                ).role
                for t in range(
                    len(
                        file_data.XFDU.metadataSection.metadataObject[
                            ind
                        ].metadataWrap.xmlData.processing.resource.processing.resource
                    )
                )
            ]
            for resource_role in resource_role_list:
                self.orbit_type = self.orbit_type_dict.get(resource_role, EOrbitType.unknown.value)
                if self.orbit_type is not EOrbitType.unknown.value:
                    break

        # Set footprint
        ind = metadata_object_list.index("measurementFrameSet")
        footprint = coda.fetch(
            file_handle,
            "XFDU/metadataSection/metadataObject["
            + str(ind)
            + "]/metadataWrap/xmlData/frameSet/frame/footPrint/coordinates",
        )
        self.footprint = np.asarray([[float(y) for y in x.split(",")] for x in footprint.split(" ")])

        # Close manifest file
        status = coda.close(file_handle)

        return True

    def __get_mid_swath_index(
        self,
    ):
        """Get mid-swath index

        :return: Mid-swath index
        :rtype: int
        """

        index_dict = {"SM": -1, "IW": 1, "EW": 2, "WV": -1}  # For SM and WV mid-swath = current swath

        return index_dict[self.acquisition_mode]

    def get_range_correction(self, swath, burst, position_range, position_azimuth, pt_geo):
        """Compute mission-dependent ALE corrections along range direction

        :param swath: Swath
        :type swath: str
        :param burst: Burst
        :type burst: int
        :param position_range: Calibration target range position []
        :type position_range: float
        :param position_azimuth: Calibration target azimuth position []
        :type position_azimuth: float
        :param pt_geo: Calibration target XYZ coordinates as numpy array of size 3
        :type pt_geo: numpy.ndarray
        :return: ALE corrections along range direction (i.e. Doppler shift correction)
        :rtype: float
        """

        # Collect useful information
        swath_index = self.get_swath_index(swath)

        pulse_bandwidth = self.pulse_list[swath_index].bandwidth
        pulse_length = self.pulse_list[swath_index].pulse_length

        t_rg = self.get_range_time_from_position(swath_index, burst, position_range)
        t_az = self.get_azimuth_time_from_position(swath_index, burst, position_azimuth)
        _, squint_frequency = self.get_squint(swath_index, burst, t_rg, t_az)

        # Compute range corrections:
        # - Doppler shift correction
        pulse_rate = pulse_bandwidth / pulse_length
        doppler_shift = squint_frequency / pulse_rate
        doppler_shift_correction = doppler_shift * cst.LIGHT_SPEED / 2

        return doppler_shift_correction

    def get_azimuth_correction(self, swath, burst, position_range, position_azimuth, pt_geo):
        """Compute mission-dependent ALE corrections along azimuth direction

        :param swath: Swath
        :type swath: str
        :param burst: Burst
        :type burst: int
        :param position_range: Calibration target range position []
        :type position_range: float
        :param position_azimuth: Calibration target azimuth position []
        :type position_azimuth: float
        :param pt_geo: Calibration target XYZ coordinates as numpy array of size 3
        :type pt_geo: numpy.ndarray
        :return: Tuple: ALE corrections along azimuth direction (i.e. bistatic delay correction, instrument timing correction, FM rate shift correction)
        :rtype: float, float, float
        """

        # Collect useful information
        swath_index = self.get_swath_index(swath)

        t_rg = self.get_range_time_from_position(swath_index, burst, position_range)
        t_az = self.get_azimuth_time_from_position(swath_index, burst, position_azimuth)
        side_looking = self.dataset_info[0].side_looking.value
        look_angle = self.general_sar_orbit[0].get_look_angle(t_az, t_rg, side_looking)
        vg = self.general_sar_orbit[0].get_velocity_ground(
            t_az, look_angle / 180 * np.pi
        )  # TODO To be used also for the computation of range and azimuth ALE instead of average value

        _, squint_frequency = self.get_squint(swath_index, burst, t_rg, t_az)

        doppler_rate = genericpoly.create_sorted_poly_list(self.dr_vector_list[swath_index]).evaluate((t_az, t_rg))
        doppler_rate_theoretical = self.get_doppler_rate_theoretical(pt_geo, t_az)

        swst_changes = self.acquisition_timeline_list[swath_index].swst_changes
        swst_index = [t < t_az for t in swst_changes[1]].index(False) - 1
        swst = swst_changes[2][swst_index]

        # Compute azimuth corrections:
        # - Bistatic delay correction (residual + improved bulk)
        mid_swath_index = self.__get_mid_swath_index()
        bistatic_delay_applied = (
            self.roi.t_rg_0[mid_swath_index, 0]
            + self.roi.n_rg[mid_swath_index, 0] / 2 * self.roi.t_rg_step[mid_swath_index, 0]
        ) / 2
        bistatic_delay = (
            -bistatic_delay_applied
            + (self.roi.t_rg_0[swath_index, burst] - position_range * self.roi.t_rg_step[swath_index, burst]) / 2
        )
        bistatic_delay_correction = -bistatic_delay * vg

        # - Instrument timing correction
        instrument_timing = swst + self.tx_pulse_latch_time
        instrument_timing_correction = instrument_timing * vg

        # - FM rate shift correction
        fmrate_shift = -squint_frequency * (
            -1 / doppler_rate + 1 / doppler_rate_theoretical
        )  # TODO Sign to be confirmed
        fmrate_shift_correction = fmrate_shift * vg

        return bistatic_delay_correction, instrument_timing_correction, fmrate_shift_correction
