#! /usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

def z_align_normal(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw of a normal vector to a quaternion and align z-axis with the normal.
    
    Args:
        roll (float): Roll angle in radians
        pitch (float): Pitch angle in radians
        yaw (float): Yaw angle in radians
    Returns:
        tuple: Quaternion (x, y, z, w)
    """

    # Align z-axis with the normal vector
    normal = np.array([roll, pitch, yaw])
    normal_normalized = normal / np.linalg.norm(normal)
    z_axis = np.array([0.0, 0.0, 1.0])
    rotation = R.align_vectors([normal_normalized], [z_axis])[0]
    quat = rotation.as_quat()
    
    return quat  # Returns (x, y, z, w)
