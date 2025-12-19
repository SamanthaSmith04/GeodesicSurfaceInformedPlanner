#! /usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_lift(pose, lift):
    """
    Calculate a lifted pose by moving along the z-axis of the oriented frame.
    Assumes Z-axis is along tool's spray axis.
    
    Args:
        pose: geometry_msgs.msg.Pose object with position and orientation
        lift (float): Distance to lift along the z-axis
    
    Returns:
        geometry_msgs.msg.Pose: New pose lifted along the z-axis
    """
    from geometry_msgs.msg import Pose
    
    if lift == 0.0:
        return pose
    
    # Create a new pose with the same orientation
    local_pose = Pose()
    local_pose.orientation = pose.orientation
    
    # Convert quaternion to scipy Rotation (note: geometry_msgs uses x,y,z,w order)
    quat = np.array([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ])
    
    # Create rotation object and normalize
    rotation = R.from_quat(quat)
    
    # Original position vector
    orig_vec = np.array([
        pose.position.x,
        pose.position.y,
        pose.position.z
    ])
    
    # Construct the liftoff vector - assumes Z-axis is along tool's spray axis
    lift_vec = np.array([0.0, 0.0, lift])
    
    # Apply rotation to lift vector and add to original position
    final_vec = orig_vec + rotation.as_matrix() @ lift_vec
    
    # Assign new position values
    local_pose.position.x = float(final_vec[0])
    local_pose.position.y = float(final_vec[1])
    local_pose.position.z = float(final_vec[2])
    
    return local_pose


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
