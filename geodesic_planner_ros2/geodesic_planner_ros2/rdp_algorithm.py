#! /usr/bin/env python3

# An altered implementation of the Ramer Douglas Peucker algorithm that allows for 
# downsampling of both position and orientation from a set of poses
#Author: Samantha Smith, smith.15485@osu.edu

import numpy as np

def rdp_run(poses, epsilon, angleThreshold):
    """
        Algorithm that implements the Ramer Douglas Peucker algorithm to downsample a set of Poses
    
        The poses are downsampled by removing points that are within a certain distance of a line between two points
        The algorithm is recursive, and will split the list of points into two subsets if a point is found that exceeds the distance threshold,
        then within those segments, the algorithm will check for points that exceed the angle threshold and if so, split the list again
        This process will continue until all points are within the thresholds
        
        Parameters:
            poses: a pose list to be processed
            epsilon: the maximum spacing between the original data points and the line between the correction points
            angleThreshold: the maximum angle between the original data points and the corrected data points about each axis
        Returns:
            results: a list of the corrected points
    """
    max_dist = 0
    first_big_rot_index = 0
    dist_index = 0
    d = 0
    #iterate over all points in the subset to find the maximum perpendicular distance and the first section where the angle is too large
    for i in range(1, len(poses)):
        d = perpendicular_distance(poses[i], poses[0], poses[-1])
        r, t = angular_distance(poses[i], poses[0])
        if d > max_dist:
            max_dist = d
            dist_index = i
        if first_big_rot_index == 0 and r >= angleThreshold:
            first_big_rot_index = i - 1 #go to the pose before to ensure that the value is less than the threshold, a new pose will be created at this positon
    #if there is a point that exceeds epsilon, split the list and run rdp on both halves
    if max_dist > epsilon: 
        poses1 = poses[:dist_index+1]
        poses2 = poses[dist_index:]
        results1 = rdp_run(poses1, epsilon, angleThreshold) #check if any points between flagged point and previous correction are within threshold
        
        results2 = rdp_run(poses2, epsilon, angleThreshold) #move onto next segment
        results = np.vstack((results1[:-1],results2)) 
    else:
        #if there is a point that exceeds the angle threshold, split the list and run rdp on both sections
        if first_big_rot_index != 0:
            poses1 = poses[:first_big_rot_index+1]
            poses2 = poses[first_big_rot_index:]

            results1 = rdp_run(poses1, epsilon, angleThreshold) #check if any points between flagged point and previous correction are within threshold
            results2 = rdp_run(poses2, epsilon, angleThreshold) #move onto next segment

            results = np.vstack((results1[:-1], results2))
        #if all points in the segment are within the thresholds, return the first and last points
        else:
            results = np.vstack((poses[0],poses[-1]))
    return results


def perpendicular_distance(current_pose, first_pose, last_pose):
    """
        Calculates the perpendicular distance between a point and a line segment in 3D space
        Parameters:
            current_pose: point to be measured
            first_pose: first point of the line segment
            last_pose: last point of the line segment
        Returns:
            out: the perpendicular distance between the point and the line segment
    """
    last_point = np.array([
        last_pose.position.x,
        last_pose.position.y,
        last_pose.position.z   
    ])

    current_point = np.array([
        current_pose.position.x,
        current_pose.position.y,
        current_pose.position.z
    ])

    first_point = np.array([
        first_pose.position.x,
        first_pose.position.y,
        first_pose.position.z
    ])

    line = np.subtract(last_point, first_point)
    if (line[0] == 0 and line[1] == 0 and line[2] == 0):
        out = 0.0
    else:
        out = np.linalg.norm(np.cross(line, np.subtract(first_point, current_point)))/np.linalg.norm(line)
    return out

def angular_distance(current_pose, first_pose):
    """
        Angular distance function
        Calculates the maximum rotation about each axis between two poses
        Parameters:
            current_pose: the current pose
            first_pose: the first pose
        Returns:
            maximum: the maximum rotation about an axis
            max_type: the axis that the maximum rotation occurs about
    """
    maximum = 0
    max_type = ""
    quat_current = np.array([
        current_pose.orientation.x,
        current_pose.orientation.y,
        current_pose.orientation.z,
        current_pose.orientation.w
    ])

    quat_first = np.array([
        first_pose.orientation.x,
        first_pose.orientation.y,
        first_pose.orientation.z,
        first_pose.orientation.w
    ])

    #normalize quaternions
    norm = np.linalg.norm(quat_current)
    quat_current /= norm
    norm = np.linalg.norm(quat_first)
    quat_first /= norm

    #convert quaternions to rotation matrices
    current = quat_to_rotation_matrix(quat_current)
    first = quat_to_rotation_matrix(quat_first)

    #get rotation of current pose relative to first pose
    relative_rotation = np.matmul(np.transpose(first), current)

    #convert to relative rotation about each axis
    rot_x = np.arctan2(relative_rotation[2][1], relative_rotation[1][1])
    rot_y = np.arctan2(-relative_rotation[2][0], np.sqrt(relative_rotation[2][1]**2 + relative_rotation[2][2]**2))
    rot_z = np.arctan2(relative_rotation[1][0], relative_rotation[0][0])

    #determine maximum rotation about an axis
    maximum = max(abs(rot_x),abs(rot_y),abs(rot_z)) 
    if (maximum == abs(rot_x)):
        max_type = "roll"
    elif (maximum == abs(rot_y)):
        max_type = "pitch"
    elif (maximum == abs(rot_z)):
        max_type = "yaw"

    return maximum, max_type

def quat_to_rotation_matrix(q):
    """
        Quaternion to rotation matrix function
        Parameters:
            q: the quaternion to be converted
        Returns:
            rotation_matrix: the rotation matrix made from q
    """
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                     [2*(x*y + z*w), 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                     [2*(x*z - y*w), 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
                    ])

