import time
import numpy as np
import geodesic_planner_ros2.rdp_algorithm as rdp_algorithm
from geometry_msgs.msg import PoseArray

def downsample(epsilon, angle_threshold, initial_poses):
    """
        Downsample function
        Reads/Generates a list of Poses to process with the rdp algorithm
        Parameters:
            epsilon: the maximum spacing between the original data points and the line between the correction points
            angle_threshold: the maximum angle between the original data points and the corrected data points about each axis
            initial_poses: a PoseArray of poses to be downsampled
        Returns:
            pose_array: a geometry_msgs.msg PoseArray of the corrected poses
    """
    messages = ""

    # print("Processing " + len(initial_poses.poses).__str__() + " points...")
    angle_threshold = np.deg2rad(angle_threshold) #convert angle threshold to radians

    #run the rdp algorithm on the poses
    corrections = rdp_algorithm.rdp_run(initial_poses.poses, epsilon, angle_threshold)
    
    #write corrected points to a pose array
    pose_array = PoseArray()
    for i in range(len(corrections)):
        pose_array.poses.append(corrections[i,0])
    
    return pose_array