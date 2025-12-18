import numpy as np
import trimesh
from trimesh.proximity import closest_point
import time

def pre_process_paths(paths, mesh, normals, max_dist=0.001):
    N = len(paths)
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



def is_point_on_part(point, mesh, tol=1e-5):
    _, dist, _ = trimesh.proximity.closest_point(mesh, [point])
    return dist[0] < tol



def pre_compute_costs(endpoints, N,  on_part):
    """
    Pre-compute squared distances from current position to all endpoints.
    
    Args:
        current_pos (np.ndarray): Current position (3,)
    """

    # cost[i, ei, j, ej]
    diff = endpoints[:, :, None, None, :] - endpoints[None, None, :, :, :]
    cost = np.linalg.norm(diff, axis=-1)

    OFF_PART_PENALTY = 1.1
    for j in range(N):
      for ej in (0, 1):
          if not on_part[j, ej]:
              cost[:, :, j, ej] *= OFF_PART_PENALTY

    return cost



def greedy_from_start(start_idx, start_ep, cost):
    N = cost.shape[0]
    visited = np.zeros(N, dtype=bool)

    order = []
    directions = []

    current_idx = start_idx
    current_ep = start_ep
    visited[current_idx] = True

    for _ in range(N - 1):
        best = (np.inf, None, None)

        for j in range(N):
            if visited[j]:
                continue

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

  # start timer
  timer = time.perf_counter()

  endpoints, on_part = pre_process_paths(paths, mesh, normals)
  N = len(paths)
  cost = pre_compute_costs(endpoints, N, on_part)


  best_total = np.inf
  best_solution = None
  print(f"pre-process time: {time.perf_counter() - timer:.4f} seconds")

  for i in range(N):
      for ei in (0, 1):
          order, dirs = greedy_from_start(i, ei, cost)

          total = 0
          cur_i, cur_e = i, ei
          for k, j in enumerate(order):
              total += cost[cur_i, cur_e, j, dirs[k]]
              cur_i, cur_e = j, 1 - dirs[k]

          if total < best_total:
              best_total = total
              best_solution = (i, ei, order, dirs)
  # end timer
  end_time = time.perf_counter()
  timer = end_time - timer
  print(f"Path ordering took {timer:.4f} seconds")
  start = (best_solution[0], best_solution[1])
  # return best_solution[2]  # return only the visit order
  ordered_paths, ordered_normals = create_new_path_ordered(
      paths, normals,
      start,
      np.array(best_solution[2]),
      endpoints,
      on_part
  )
  return np.array(best_solution[2]), ordered_paths, ordered_normals

def create_new_path_ordered(paths, normals, start, visit_order, endpoints, on_part):
    ordered_paths = []
    ordered_normals = []

    N = visit_order.shape[0]
    first = True
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
            if len(current_path) > 0:
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