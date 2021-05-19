# -*- coding: utf-8 -*-

import numpy as np

from arepytools.geometry.wgs84 import _A_MAX, _A_MIN


def get_zerodoppler_reference_frame(pos, vel, plane_flag=0):

    # Check inputs
    if not (pos.size == 3 and vel.size == 3):
        raise RuntimeError("Input pos and vel must be 3-elements vectors")
    if pos.shape[0] == 1:
        pos.shape = (3, 1)
    if vel.shape[0] == 1:
        vel.shape = (3, 1)

    # Build reference frame
    # - x-versor oriented as sensor non-inertial velocity
    x = vel / np.linalg.norm(vel)

    # - y-versor given by the cross product between x and sensor position corrected with Earth eccentricity
    if not plane_flag:
        beta_mat = np.zeros((3, 3))
        beta_mat[2, 2] = 0.0060611
        pos = pos + np.matmul(beta_mat, pos)
    temp = np.cross(x, pos, axis=0)
    y = temp / np.linalg.norm(temp)

    # - z-versor completing the reference frame
    temp = np.cross(x, y, axis=0)
    z = temp / np.linalg.norm(temp)

    reference_frame = np.concatenate((x, y, z), axis=1)

    return reference_frame


def find_ellipsoid_intersection__DEPRECATED(x, pos, delta_height=0.0):

    x = x.squeeze()
    pos = pos.squeeze()

    # Intersection with ellipsoid: (x^2+y^2)/AMAX^2 + z^2 / AMIN^2 = 1
    # This is obtained by imposing that the vector (pos-k*x) belongs to the ellipsoid

    # Simul a difference in the height
    AMAX = _A_MAX + delta_height
    AMIN = _A_MIN + delta_height
    a = (x[0] ** 2 + x[1] ** 2) / AMAX ** 2 + x[2] ** 2 / AMIN ** 2
    b = -(2 * pos[0] * x[0] + 2 * pos[1] * x[1]) / AMAX ** 2 - 2 * pos[2] * x[2] / AMIN ** 2
    c = (pos[0] ** 2 + pos[1] ** 2) / AMAX ** 2 + pos[2] ** 2 / AMIN ** 2 - 1

    # Find the modulus of x
    k = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    pp = pos - k * x

    return k, pp


def find_ellipsoid_intersection(x, pos, semi_axis_major=_A_MAX, semi_axis_minor=_A_MIN, delta_height=0.0):

    # The following system of equations is solved:
    #   Rect in space (parametric form): pos(k) = pos + k*x
    #   Ellipsoid                      : (pos(1)+pos(2))^2/semi_axis_major^2+pos(3)^2/semi_axis_minor^2=1
    #
    #   The parameter k for which the point of the rect belongs to the ellipsoid is
    #   derived solving a second degree equation. Three cases are possible:
    #       1) Two intersections bewteen LOS and ellipsoid
    #       2) One intersection bewteen LOS and ellipsoid
    #       3) No intersection bewteen LOS and ellipsoid

    # Normalize Line-of-Sight
    x = (x / np.linalg.norm(x)).squeeze()
    pos = pos.squeeze()

    # Apply delta height
    AMAX = semi_axis_major + delta_height
    AMIN = semi_axis_minor + delta_height

    # Second degree equation coefficients
    a = (x[0] ** 2 + x[1] ** 2) / AMAX ** 2 + (x[2] ** 2) / AMIN ** 2
    b = (2 * pos[0] * x[0] + 2 * pos[1] * x[1]) / AMAX ** 2 + 2 * pos[2] * x[2] / AMIN ** 2
    c = (pos[0] ** 2 + pos[1] ** 2) / AMAX ** 2 + pos[2] ** 2 / AMIN ** 2 - 1

    # Find solutions
    eq_det = b ** 2 - 4 * a * c
    if eq_det < 0:
        # No solution: LOS points ouside ellipsoid
        raise ValueError("No intersection between ellipsoid and Line-of-Sight.")
    elif eq_det == 0:
        # One solution: horizon point
        pos = pos - b / (2 * a) * x
    else:
        # Two solutions: the correct one is the nearest (farthest one is not in visibility)
        v1 = (-b + np.sqrt(eq_det)) / (2 * a) * x
        v2 = (-b - np.sqrt(eq_det)) / (2 * a) * x
        k1 = v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2
        k2 = v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2
        if k1 < k2:
            k = -v1[0] / x[0]
            pos = pos + v1
        else:
            k = -v2[0] / x[0]
            pos = pos + v2

    return k, pos


def get_rotation_matrix(yaw, pitch, roll, rotation_order):

    # Convert angle to radians
    yaw = yaw / 180 * np.pi
    pitch = pitch / 180 * np.pi
    roll = roll / 180 * np.pi

    # Compute rotation matrices
    rot_mat_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    rot_mat_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rot_mat_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    # Combine rotation matrices
    if rotation_order == "YPR":
        rot_mat = np.matmul(rot_mat_yaw, np.matmul(rot_mat_pitch, rot_mat_roll))
    elif rotation_order == "YRP":
        rot_mat = np.matmul(rot_mat_yaw, np.matmul(rot_mat_roll, rot_mat_pitch))
    elif rotation_order == "PYR":
        rot_mat = np.matmul(rot_mat_pitch, np.matmul(rot_mat_yaw, rot_mat_roll))
    elif rotation_order == "PRY":
        rot_mat = np.matmul(rot_mat_pitch, np.matmul(rot_mat_roll, rot_mat_yaw))
    elif rotation_order == "RYP":
        rot_mat = np.matmul(rot_mat_roll, np.matmul(rot_mat_yaw, rot_mat_pitch))
    elif rotation_order == "RPY":
        rot_mat = np.matmul(rot_mat_roll, np.matmul(rot_mat_pitch, rot_mat_yaw))
    else:
        raise ValueError("Invalid value for rotation order: {}".format(rotation_order))

    return rot_mat


def look_to_slant_range(pos, vel, look_angle, reference_frame, delta_height=0.0, aarf=None):

    if np.max(np.abs(look_angle) > np.pi):
        raise ValueError(
            "Invalid value for look angle: {}. Check if degrees have been used in place of radians".format(look_angle)
        )
    if reference_frame is None:
        reference_frame = "ZERODOPPLER"

    if reference_frame == "ZERODOPPLER":
        vel_non_in = vel
        nom_ref = get_zerodoppler_reference_frame(pos, vel_non_in)
        h_sat, _ = find_ellipsoid_intersection(-nom_ref[:, 2], pos)
    elif reference_frame == "GEOCENTRIC" or reference_frame == "GEODETIC":
        raise NotImplementedError  # TODO
    else:
        raise ValueError("Invalid value for system reference frame: {}".format(reference_frame))

    AMAX = _A_MAX + delta_height
    AMIN = _A_MIN + delta_height

    n = look_angle.size
    slant_range = np.zeros((1, n))
    incidence_angle = np.zeros((1, n))
    ground_point = np.zeros((3, n))

    if aarf is None:
        aarf = nom_ref
        aarf_look = 0.0
    else:
        aarf_look = np.arccos(np.dot(nom_ref[:, 2], aarf[:, 2]))
    look_angle = look_angle - aarf_look

    for i in range(n):
        # 1. Rotates the geodetic/geocentric reference frame
        nominal_rot_mat = get_rotation_matrix(0, 0, -look_angle[i] * 180 / np.pi, "YPR")
        # Apply nominal attitude
        # The columns vectors of dogcrf are the versors of the desired platform attitude
        dogcrf = (np.matmul(nominal_rot_mat.transpose(), aarf.transpose())).transpose()

        # 2. Determines the point on Earth as intersection with the ellipsoid of the z axis
        slant_range[0, i], P = find_ellipsoid_intersection(-dogcrf[:, 2], pos, delta_height=delta_height)
        ground_point[:, i] = P

        # 3. Considering the point under the satellite P0 and the observed point on Earth P, determines the incidence angle
        v_norm = np.array([[2 * P[0] / AMAX ** 2], [2 * P[1] / AMAX ** 2], [2 * P[2] / AMIN ** 2]])  # exiting normal
        point_to_sat = pos.squeeze() - P  # vector from the point on Earth to the satellite
        incidence_angle[0, i] = np.arccos(
            np.dot(point_to_sat, v_norm / np.linalg.norm(point_to_sat) / np.linalg.norm(v_norm))
        )

    return slant_range, incidence_angle, h_sat, ground_point, aarf_look
