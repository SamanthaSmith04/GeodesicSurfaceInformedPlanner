"""
Path ordering functions to order a set of path segments to minimize travel distance between them.
Uses a greedy nearest-neighbor heuristic.
"""

import numpy as np
import trimesh
from trimesh.proximity import closest_point
import time

def pre_process_paths(paths, mesh, normals, max_dist=0.001):
    """
    For each path segment, get the endpoints and check if they are pointing towards the mesh surface.
    Args:
        paths (list of np.ndarray): List of path segments
        mesh (trimesh.Trimesh): Mesh of the part
        normals (list of np.ndarray): List of normal vectors for each path segment
        max_dist (float): Maximum distance to consider a ray hit as valid
    Returns:
        endpoints (np.ndarray): Array of shape (N, 2, 3) with endpoints of each path segment
        on_part (np.ndarray): Boolean array of shape (N, 2) indicating if endpoints are on the part
    """
    
    endpoints = np.array([[seg[0], seg[-1]] for seg in paths])
    flat = endpoints.reshape(-1, 3)

    spray_dirs = np.array([[norm[0], norm[-1]] for norm in normals]).reshape(-1, 3)

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=flat,
        ray_directions=spray_dirs,
        multiple_hits=False
    )

    num_on_part = 0
    on_part = np.zeros(len(flat), dtype=bool)
    for i, ray_idx in enumerate(index_ray):
        dist = np.linalg.norm(locations[i] - flat[ray_idx])
        if dist <= max_dist:
            on_part[ray_idx] = True
            num_on_part += 1

    print(f"Endpoints on part: {num_on_part} / {len(flat)}")

    return endpoints, on_part.reshape(-1, 2)


def pre_compute_costs(endpoints, N,  on_part):
    """
    Pre-compute squared distances from current position to all endpoints.
    
    Args:
        current_pos (np.ndarray): Current position (3,)
    """

    # cost[i, ei, j, ej] = cost from endpoint ei of segment i to endpoint ej of segment j
    diff = endpoints[:, :, None, None, :] - endpoints[None, None, :, :, :]
    cost = np.linalg.norm(diff, axis=-1)

    OFF_PART_PENALTY = 1.1 # penalty for going to an endpoint that is not on the part, prioritizes staying on the part
    for j in range(N):
      for ej in (0, 1):
          if not on_part[j, ej]:
              cost[:, :, j, ej] *= OFF_PART_PENALTY

    return cost



def greedy_from_start(start_idx, start_ep, cost):
    """
    Greedy nearest-neighbor path ordering heuristic
    Args:
        start_idx (int): Starting segment index
        start_ep (int): Starting endpoint index (0 or 1)
        cost (np.ndarray): Pre-computed cost array of shape (N, 2, N, 2)
    Returns:
        order (list): Ordered list of segment indices
        directions (list): List of endpoint indices (0 or 1) for each segment in the order
    """
    N = cost.shape[0]
    visited = np.zeros(N, dtype=bool)

    order = [start_idx]
    directions = [start_ep]

    current_idx = start_idx
    current_ep = start_ep
    visited[current_idx] = True

    # check all remaining segments
    for _ in range(N - 1):
        best = (np.inf, None, None)
        # find nearest unvisited segment
        for j in range(N):
            if visited[j]:
                continue

            # find best endpoint to approach
            for ej in (0, 1):
                c = cost[current_idx, current_ep, j, ej]
                if c < best[0]:
                    best = (c, j, ej)

        _, next_idx, next_ep = best
        visited[next_idx] = True

        order.append(next_idx)
        directions.append(next_ep)

        current_idx = next_idx
        current_ep = 1 - next_ep  # exit from the other end

    return order, directions



def order_paths(paths, mesh, normals):
    """
    Orders the path segments to minimize travel distance between them using a greedy nearest-neighbor heuristic.
    Args:
        paths (list of np.ndarray): List of path segments
        mesh (trimesh.Trimesh): Mesh of the part
        normals (list of np.ndarray): List of normal vectors for each path segment
    Returns:
        visit_order (np.ndarray): Order in which to visit segments
        ordered_paths (list of np.ndarray): Ordered list of path segments
        ordered_normals (list of np.ndarray): Ordered list of normal vectors
    """

    # get endpoint positions and whether they are pointing over the part or not
    endpoints, on_part = pre_process_paths(paths, mesh, normals)
    N = len(paths)
    # calculate costs from all endpoints to eachother
    cost = pre_compute_costs(endpoints, N, on_part)

    best_total = np.inf
    best_solution = None
    all_on_part = np.all(on_part)

    for i in range(N): # check each segment
        for ei in (0, 1): # check each endpoint
            # ensure first endpoint is not on the part
            if (on_part[i][ei]) and not all_on_part:
                continue
            # get best cost path from start point
            order, dirs = greedy_from_start(i, ei, cost)

            total = 0
            cur_i, cur_e = i, ei
            # calculate total cost
            for k, j in enumerate(order):
                total += cost[cur_i, cur_e, j, dirs[k]]
                cur_i, cur_e = j, 1 - dirs[k]

            # store best cost so far
            if total < best_total:
                best_total = total
                best_solution = (i, ei, order, dirs)

    # store starting position
    start = (best_solution[0], best_solution[1])
    
    # store the new paths in order
    ordered_paths, ordered_normals = create_new_path_ordered(
        paths, 
        normals,
        start,
        np.array(best_solution[2]), #ordered segments
        endpoints,
        on_part
    )
    return np.array(best_solution[2]), ordered_paths, ordered_normals

def create_new_path_ordered(paths, normals, start, visit_order, endpoints, on_part):
    """
    Orders the path segments based on their visit order and start position.
    Reverses paths when endpoints are visited in opposite order.
    If the next path segment starts on the part, it is appended to the current path, to ensure the spray stays on during the transition.
    Args:
        paths (list of np.ndarray): List of path segments
        normals (list of np.ndarray): List of normal vectors for each path segment
        start (tuple): Starting segment index and endpoint (index 0 or 1)
        visit_order (np.ndarray): Order in which to visit segments
        endpoints (np.ndarray): Endpoints of each path segment
        on_part (np.ndarray): Boolean array indicating if endpoints are on the part
    Returns:
        ordered_paths (list of np.ndarray): Ordered list of path segments
        ordered_normals (list of np.ndarray): Ordered list of normal vectors
    """
    ordered_paths = []
    ordered_normals = []

    print(f"visit order length: {len(visit_order)}")

    first = True
    # reverse any paths where endpoint1 is closer to the last endpoint of the previous path
    for idx in visit_order:
        if first:
            first = False
            # orient first path according to start
            path = paths[idx]
            normal = normals[idx]
            if start[1] == 1:
                paths[idx] = path[::-1]
                normals[idx] = normal[::-1]
                on_part[idx] = on_part[idx][::-1]
                endpoints[idx] = endpoints[idx][::-1]
            continue
        # reverse sections if needed
        path = paths[idx]
        normal = normals[idx]
        if (np.linalg.norm(endpoints[visit_order[visit_order.tolist().index(idx)-1], -1] - endpoints[idx, 0]) > np.linalg.norm(endpoints[visit_order[visit_order.tolist().index(idx)-1], -1] - endpoints[idx, 1])):
            paths[idx] = path[::-1]
            normals[idx] = normal[::-1]
            on_part[idx] = on_part[idx][::-1]
            endpoints[idx] = endpoints[idx][::-1]

    # order paths.
    # if the next path is on the surface, it needs to be appended to the current path
    current_path = []
    current_normal = []
    for idx in visit_order:
        if on_part[idx][0] or len(current_path) == 0: # start point is on the part (spray gun must already be on)
            current_path.extend(paths[idx])
            current_normal.extend(normals[idx])
        else:
            if len(current_path) > 0 or len(ordered_paths) == 0:
                ordered_paths.append(np.array(current_path))
                ordered_normals.append(np.array(current_normal))
            current_normal = []
            current_path = []
            current_path.extend(paths[idx])
            current_normal.extend(normals[idx])

    if len(current_path) > 0:
        ordered_paths.append(np.array(current_path))
        ordered_normals.append(np.array(current_normal))
            
    return ordered_paths, ordered_normals