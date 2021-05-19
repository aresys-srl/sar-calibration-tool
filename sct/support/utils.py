# -*- coding: utf-8 -*-

import numpy as np


def compute_angle_between_vectors(a, b):

    a_norm = a / np.linalg.norm(a, axis=0)
    b_norm = b / np.linalg.norm(b, axis=0)
    dot_product = np.sum((a_norm * b_norm), axis=0)
    dot_product = np.where((dot_product < -1), -1, dot_product)
    dot_product = np.where((dot_product > 1), 1, dot_product)
    angle = np.arccos(dot_product)

    return angle


def compute_incidence_angle(sat_xyz_coordinates, xyz_coordinates):

    los_target_to_sat = sat_xyz_coordinates - xyz_coordinates
    incidence_angle = compute_angle_between_vectors(los_target_to_sat, xyz_coordinates)

    return incidence_angle
