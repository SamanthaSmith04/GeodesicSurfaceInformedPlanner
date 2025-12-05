import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.interpolate import UnivariateSpline, Rbf
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import networkx as nx

def fit_parametric_curve(points, smoothing=0.0):
    """
    Fit a parametric curve to a set of 3D points using spline fitting.
    Args:
        points (list of np.array): List of 3D points to fit the curve to.
        smoothing (float): Smoothing factor for the spline.
    Returns:
        spline_x, spline_y, spline_z: Spline functions for x, y, z coordinates.
    """
    points = order_points_nearest_neighbor(points)
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
    # ax.scatter(points[0, 0], points[0, 1], points[0, 2], color='g', s=100, label='Start Point')
    # t_fit = np.linspace(0, 1, 100)
    # ax.plot(spline_x(t_fit), spline_y(t_fit), spline_z(t_fit), color='r', label='Fitted Curve')

    # # ax.set_xlim(0.0, 1.5)
    # # ax.set_ylim(0.0, 1.5)
    # # ax.set_zlim(0.0, 1.5)
  
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')  
    # plt.title('Parametric Curve Fitting')
    # plt.legend()
    # plt.show()

    return spline_x, spline_y, spline_z
    

def order_points_nearest_neighbor(points):
    """
    Order a list of 3D points to produce the shortest path using MST traversal.
    Args:
        points (list of np.array): List of 3D points to order.
    Returns:
        ordered_points (list of np.array): Ordered list of 3D points.
    """
    if len(points) == 0 or len(points) == 1:
        return points
    
    points = np.array(points)
    
    # Build complete graph with edge weights as distances
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            G.add_edge(i, j, weight=dist)
    
    # Build MST for approximate shortest path
    mst = nx.minimum_spanning_tree(G)

    # Find the two leaves (endpoints) of the MST
    leaves = [node for node in mst.nodes() if mst.degree(node) == 1]
    print(f"leaves len: {len(leaves)}")
    
    # Pick the leaf farthest from centroid as starting point
    centroid = points.mean(axis=0)
    leaf_coords = points[leaves]
    dists_to_centroid = np.linalg.norm(leaf_coords - centroid, axis=1)
    start_leaf = leaves[np.argmax(dists_to_centroid)]
    
    # Traverse the MST using DFS to get ordered path
    ordered_indices = list(nx.algorithms.traversal.depth_first_search.dfs_preorder_nodes(mst, source=start_leaf))
    
    # Return points in the ordered sequence
    return points[ordered_indices]


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

def straight_line_interpolation(start_point, start_normal, end_point, end_normal, point_spacing):
    """
    Interpolate points along a straight line between two 3D points.
    Args:
        start_point (np.array): Starting 3D point.
        start_normal (np.array): Normal at the starting point.
        end_point (np.array): Ending 3D point.
        end_normal (np.array): Normal at the ending point.
        point_spacing (float): Desired spacing between interpolated points.
    Returns:
        interpolated_points (list of np.array): List of interpolated 3D points.
        interpolated_normals (list of np.array): List of normals at the interpolated points.
    """
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    direction = end_point - start_point
    length = np.linalg.norm(direction)
    if length <= point_spacing:
        return [start_point], [start_normal]
    
    direction = direction / length

    num_points = int(length / point_spacing) + 1
    if num_points < 2:
        return [start_point], [start_normal]
    interpolated_points = [start_point + direction * point_spacing * i for i in range(num_points-1)]
    interpolated_normals = [start_normal + (end_normal - start_normal) * (i / (num_points - 1)) for i in range(num_points-1)]

    return interpolated_points, interpolated_normals

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
    # pruned_points = [interpolated_points[0]]
    # for point in interpolated_points[1:]:
    #     if np.linalg.norm(point - pruned_points[-1]) >= point_spacing * 0.05:
    #         pruned_points.append(point)
    # interpolated_points = pruned_points

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

def extrapolate_endpoints(points, normals, extension_length=0.05, point_spacing=0.01):
    """
    Extrapolate the endpoints of a list of 3D points along the direction of the endpoints. (Normal directions will be the same as the original endpoints)
    This will add evenly spaced points along the direction of the endpoints using the same spacing method as the original points.
    Args:
        points (list of np.array): List of 3D points.
        normals (list of np.array): List of normals at the points.
        extension_length (float): Length to extend at each endpoint.
    Returns:
        extended_points (list of np.array): List of 3D points with extrapolated endpoints.
        extended_normals (list of np.array): List of normals at the extended points.
    """ 
    if len(points) < 2:
        return points, normals
    
    # check if starting and ending segments are too close together (loop)
    if np.linalg.norm(points[1] - points[0]) < point_spacing * 0.5 or np.linalg.norm(points[-1] - points[-2]) < point_spacing * 0.5:
        return points, normals

    # compute spacing based on first two points
    start_dir = None
    if len(points) < 4:
        start_dir = points[1] - points[0]
    else:
        # get average direction over first 4 points
        start_dir = (points[1] - points[0] + points[2] - points[1] + points[3] - points[2]) / 3.0
    start_spacing = point_spacing
    start_dir = start_dir / np.linalg.norm(start_dir)   
    num_start_points = int(extension_length / start_spacing)
    extended_start_points = [points[0] - start_dir * start_spacing * (i + 1) for i in range(num_start_points)][::-1]
    extended_start_normals = [normals[0] for _ in range(num_start_points)]
    # compute spacing based on last two points
    end_dir = None
    if len(points) < 4:
        end_dir = points[-1] - points[-2]
    else:
        # get average direction over last 4 points
        end_dir = (points[-1] - points[-2] + points[-2] - points[-3] + points[-3] - points[-4]) / 3.0
    end_spacing = point_spacing
    end_dir = end_dir / np.linalg.norm(end_dir)
    num_end_points = int(extension_length / end_spacing)
    extended_end_points = [points[-1] + end_dir * end_spacing * (i + 1) for i in range(num_end_points)]    
    extended_end_normals = [normals[-1] for _ in range(num_end_points)]
    extended_points = extended_start_points + points + extended_end_points
    extended_normals = extended_start_normals + normals + extended_end_normals
    return extended_points, extended_normals