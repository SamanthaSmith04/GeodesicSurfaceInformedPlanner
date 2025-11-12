import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

def fit_parametric_curve(points, smoothing=0.0):
    """
    Fit a parametric curve to a set of 3D points using spline fitting.
    Args:
        points (list of np.array): List of 3D points to fit the curve to.
        smoothing (float): Smoothing factor for the spline.
    Returns:
        spline_x, spline_y, spline_z: Spline functions for x, y, z coordinates.
    """
    points = np.array(points)
    t = np.linspace(0, 1, len(points))

    if len(points) <= 3:
        # fall back to linear fit
        # print("Not enough points for spline fitting, falling back to linear fit.")
        k = min(3, len(points)-1)  # spline degree cannot exceed m-1
        spline_x = UnivariateSpline(t, points[:, 0], s=smoothing, k=k)
        spline_y = UnivariateSpline(t, points[:, 1], s=smoothing, k=k)
        spline_z = UnivariateSpline(t, points[:, 2], s=smoothing, k=k)
        return spline_x, spline_y, spline_z

    spline_x = UnivariateSpline(t, points[:, 0], s=smoothing)
    spline_y = UnivariateSpline(t, points[:, 1], s=smoothing)
    spline_z = UnivariateSpline(t, points[:, 2], s=smoothing)

    # # Plot the fitted curve
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Data Points')
    # t_fit = np.linspace(0, 1, 100)
    # ax.plot(spline_x(t_fit), spline_y(t_fit), spline_z(t_fit), color='r', label='Fitted Curve')

    # ax.set_xlim(0.0, 1.5)
    # ax.set_ylim(0.0, 1.5)
    # ax.set_zlim(0.0, 1.5)
  
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')  
    # plt.title('Parametric Curve Fitting')
    # plt.legend()
    # plt.show()

    return spline_x, spline_y, spline_z
    

def order_points_nearest_neighbor(points):
    """
    Order a list of 3D points using the nearest neighbor approach.
    Args:
        points (list of np.array): List of 3D points to order.
    Returns:
        ordered_points (list of np.array): Ordered list of 3D points.
    """
    if len(points) == 0 or len(points) == 1:
        return points
    

    ordered_points = []

    points_set = []
    for p in points:
        points_set.append(tuple(p))

    # find point with the fewest neighbors as starting point
    start_idx = 0
    min_neighbors = float('inf')
    for i, point in enumerate(points_set):
        count = 0
        for other_point in points_set:
            if point != other_point:
                dist = np.linalg.norm(np.array(point) - np.array(other_point))
                if dist < 0.1:
                    count += 1
        if count < min_neighbors:
            min_neighbors = count
            start_idx = i

    ordered_points.append(list(points_set[start_idx]))
    current_point = points_set[start_idx]
    while points_set:
        nearest_point = min(points_set, key=lambda p: np.linalg.norm(np.array(current_point) - np.array(p)))
        ordered_points.append(list(nearest_point))
        points_set.remove(nearest_point)
        current_point = nearest_point

    return ordered_points


def split_isoline_gaps(isoline, max_gap=0.1):
    """
    Split an isoline into segments based on a maximum gap distance. Reduces large gaps in the isoline.
    Args:
        isoline (list of np.array): List of 3D points representing the isoline.
        max_gap (float): Maximum allowed gap distance between consecutive points.
    Returns:
        segments (list of list of np.array): List of isoline segments.
    """
    isoline = order_points_nearest_neighbor(isoline)
    if len(isoline) == 0:
        return []
    isoline = [np.array(p) for p in isoline]

    segments = []
    current_segment = [isoline[0]]

    for i in range(1, len(isoline)):
        dist = np.linalg.norm(np.array(isoline[i]) - np.array(isoline[i - 1]))
        if dist <= max_gap:
            current_segment.append(isoline[i])
        else:
            if len(current_segment) > 0:
                segments.append(current_segment)
            current_segment = [isoline[i]]

    if len(current_segment) > 0:
        segments.append(current_segment)

    return segments



def interpolate_path(x_interp, y_interp, z_interp, point_spacing, mesh):
    """
    Interpolate points along a parametric curve defined by spline functions.
    Args:
        x_interp, y_interp, z_interp: Spline functions for x, y, z coordinates.
        point_spacing (float): Desired spacing between interpolated points.
        mesh (trimesh.Trimesh): Mesh to get normals from.
    Returns:
        interpolated_points (list of np.array): List of interpolated 3D points.
        interpolated_normals (list of np.array): List of normals at the interpolated points.
    """

    t_values = np.linspace(0, 1, 10000)
    x_vals = x_interp(t_values)
    y_vals = y_interp(t_values)
    z_vals = z_interp(t_values)

    interpolated_points = []
    interpolated_normals = []

    current_point = np.array([x_vals[0], y_vals[0], z_vals[0]])
    x_vals = x_vals[1:]
    y_vals = y_vals[1:]
    z_vals = z_vals[1:]
    current_dist = 0.0
    interpolated_points.append(current_point)
    
    last_step = np.array([x_vals[0], y_vals[0], z_vals[0]])
    for i in range(1,len(x_vals)):
        step_point = np.array([x_vals[i], y_vals[i], z_vals[i]])
        diff = np.linalg.norm(step_point - last_step)
        current_dist += diff
        if current_dist >= point_spacing:
            interpolated_points.append(step_point)
            current_dist = 0.0
        last_step = step_point
        
    interpolated_normals = get_normals(interpolated_points, mesh)

    return interpolated_points, interpolated_normals


def get_normals(points, mesh):
    """
    Get normals from mesh for a list of points.
    Args:
        points (list of np.array): List of 3D points.
        mesh (trimesh.Trimesh): Mesh to get normals from.
    Returns:
        normals (list of np.array): List of normals at the points.
    """
    normals = []
    for point in points:
        closest, dist, face_idx = trimesh.proximity.closest_point(mesh, point.reshape(1, 3))
        normal = mesh.face_normals[mesh.faces.tolist().index(mesh.faces[face_idx[0]].tolist())]
        normals.append(normal)
    return normals