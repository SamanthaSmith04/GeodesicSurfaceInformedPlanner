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
    print(f"Ordering {len(points_set)} points using nearest neighbor.")

    # find point with the fewest neighbors as starting point
    tree = cKDTree(points_set)
    min_neighbors = float('inf')
    start_idx = 0
    for i, point in enumerate(points_set):
        neighbors = tree.query_ball_point(point, r=0.1)
        if len(neighbors) < min_neighbors:
            min_neighbors = len(neighbors)
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

    # densely sample the curve to compute arc length
    t_dense = np.linspace(0, 1, 100000)
    x_dense = x_interp(t_dense)
    y_dense = y_interp(t_dense)
    z_dense = z_interp(t_dense)
    dense_points = np.vstack((x_dense, y_dense, z_dense)).T

    segment_lengths = np.linalg.norm(np.diff(dense_points, axis=0), axis=1)
    arc_lengths = np.cumsum(np.insert(segment_lengths, 0, 0))
    total_length = arc_lengths[-1]
    num_points = int(total_length / point_spacing) + 1
    target_arc_lengths = np.linspace(0, total_length, num_points)

    # get uniformly spaced t values based on arc length
    t_uniform = np.interp(target_arc_lengths, arc_lengths, t_dense) 

    x_vals = x_interp(t_uniform)
    y_vals = y_interp(t_uniform)
    z_vals = z_interp(t_uniform)
    interpolated_points = [np.array([x, y, z]) for x, y, z in zip(x_vals, y_vals, z_vals)]

    # prune any points in the path that are too close to eachother (regardless of ordering)
    pruned_points = [interpolated_points[0]]
    for point in interpolated_points[1:]:
        if np.linalg.norm(point - pruned_points[-1]) >= point_spacing * 0.5:
            pruned_points.append(point)
    interpolated_points = pruned_points

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