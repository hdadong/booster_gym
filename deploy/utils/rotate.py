import numpy as np


def rotate_vector_inverse_rpy(roll, pitch, yaw, vector):
    """
    Rotate a vector by the inverse of the given roll, pitch, and yaw angles.

    Parameters:
    roll (float): The roll angle in radians.
    pitch (float): The pitch angle in radians.
    yaw (float): The yaw angle in radians.
    vector (np.ndarray): The 3D vector to be rotated.

    Returns:
    np.ndarray: The rotated 3D vector.
    """
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return (R_z @ R_y @ R_x).T @ vector


def rotate_vector_rpy(roll, pitch, yaw, vector):
    """
    Rotate a vector by the inverse of the given roll, pitch, and yaw angles.

    Parameters:
    roll (float): The roll angle in radians.
    pitch (float): The pitch angle in radians.
    yaw (float): The yaw angle in radians.
    vector (np.ndarray): The 3D vector to be rotated.

    Returns:
    np.ndarray: The rotated 3D vector.
    """
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return (R_z @ R_y @ R_x) @ vector

def rpy_zyx_to_quat_wxyz(roll, pitch, yaw):
    """
    Convert intrinsic ZYX Euler angles (roll=x, pitch=y, yaw=z) to quaternion [w, x, y, z].
    Angles are in radians.
    """
    cr, sr = np.cos(roll/2.0),  np.sin(roll/2.0)
    cp, sp = np.cos(pitch/2.0), np.sin(pitch/2.0)
    cy, sy = np.cos(yaw/2.0),   np.sin(yaw/2.0)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def rpy_xyz_to_quat_wxyz(roll, pitch, yaw):
    """
    Convert intrinsic XYZ Euler angles (roll=x, pitch=y, yaw=z)
    to quaternion [w, x, y, z]. Angles in radians.
    """
    cr, sr = np.cos(roll/2.0),  np.sin(roll/2.0)
    cp, sp = np.cos(pitch/2.0), np.sin(pitch/2.0)
    cy, sy = np.cos(yaw/2.0),   np.sin(yaw/2.0)

    w = cr*cp*cy - sr*sp*sy
    x = sr*cp*cy + cr*sp*sy
    y = cr*sp*cy - sr*cp*sy
    z = cr*cp*sy + sr*sp*cy
    return np.array([w, x, y, z])
    