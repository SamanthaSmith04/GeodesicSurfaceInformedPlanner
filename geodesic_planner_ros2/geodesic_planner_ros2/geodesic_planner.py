#!/usr/bin/env python3

import numpy as np
import potpourri3d as pp3d # https://pypi.org/project/potpourri3d/
import trimesh
import matplotlib.pyplot as plt

'''
  Computes geodesic isolines for the entire surface
'''
class GeodesicPlanner:
    def __init__(self, mesh_file):
        self.mesh = trimesh.load(mesh_file)
        
        # Debug: Print mesh information
        print(f"Mesh loaded successfully!")
        print(f"Number of vertices: {len(self.mesh.vertices)}")
        print(f"Number of faces: {len(self.mesh.faces)}")
        print(f"Mesh bounds: {self.mesh.bounds}")
        
        # TODO: parameterize this later
        # self.source_vertex = np.array(self.mesh.centroid)  # Use the centroid of the mesh as the source vertex
        # self.source_vertex[2] = 0.5
        self.source_vertex = np.array([0.5, 0.5, 0.5])  # Example hardcoded source vertex
        # self.source_vertex = np.array([0, 0, 0])  # Example hardcoded source vertex
        self.solver = pp3d.MeshHeatMethodDistanceSolver(self.mesh.vertices, self.mesh.faces)


    def compute_geodesic_distances(self, source_vertex_indices=None) -> np.ndarray:
        '''
        Compute geodesic distances from the source vertex to all other vertices in the mesh.
        Parameters:
            source_vertex_indices: Optional index of the source vertices in the mesh. If None, uses the closest vertex to self.source_vertex.
        Returns:
            distances: A numpy array of geodesic distances from the source vertex to all other vertices in the mesh.
        '''
        # find the closest vertex in the mesh to the source vertex
        # source_index = self.mesh.kdtree.query(self.source_vertex)[1]
        # print("Actual source vertex point:", self.source_vertex)
        # print("Source vertex index:", source_index)
        # print("Closest vertex coordinates:", self.mesh.vertices[source_index])
        # print("Distance between source point and closest vertex:", np.linalg.norm(self.mesh.vertices[source_index] - self.source_vertex))

        source_indices = []
        for idx in source_vertex_indices:
            point = np.array([idx.x, idx.y, idx.z])
            print(f"Processing source vertex index: {idx}")
            source_index = self.mesh.kdtree.query(point)[1]
            source_indices.append(source_index)
            print(f"Source vertex index {idx} coordinates:", self.mesh.vertices[source_index])
            print(f"Distance between source point and vertex {idx}:", np.linalg.norm(self.mesh.vertices[source_index] - self.source_vertex))

            # Check if source index is valid
            if source_index >= len(self.mesh.vertices) or source_index < 0:
                print(f"ERROR: Invalid source index {source_index}")
                return None
            
        # compute geodesic distances from the source vertex to all other vertices
        distances = []
        try:
            if len(source_indices) == 1:
                source_index = source_indices[0]
                distances = self.solver.compute_distance(source_index)
            else:
                distances = self.solver.compute_distance_multisource(source_indices)
        except Exception as e:
            print(f"ERROR in geodesic computation: {e}")
            return None
        
        # Debug: Print some statistics about the distances
        print(f"Min distance: {np.min(distances)}")
        print(f"Max distance: {np.max(distances)}")
        print(f"Mean distance: {np.mean(distances)}")
        print(f"Number of unique distances: {len(np.unique(distances))}")
        
        return distances
    
    def find_nearest_sources(self, points: list) -> list:
        """
        Find the nearest vertices in the mesh for the given list of points.
        Parameters:
            points: A list of Point objects representing source points.
        Returns:
            source_points: A list of nearest vertex coordinates in the mesh.
        """
        source_points = []
        for idx in points:
            point = np.array([idx.x, idx.y, idx.z])
            source_index = self.mesh.kdtree.query(point)[1]
            source_points.append(self.mesh.vertices[source_index])
        return source_points
    

    def find_single_isoline(self, target_distance : float, distances : np.ndarray) -> (list, list):
        '''
        Find the vertices that lie on the isoline at the specified distance.
        Parameters:
            target_distance: The distance at which to find the isoline.
            distances: A numpy array of geodesic distances from the source vertex to all other vertices in the mesh.
        Returns:
            isoline_vertices: A list of vertices that lie on the isoline at the specified distance
            normal_vectors: A list of normal vectors of the faces corresponding to the isoline vertices
        '''
        if distances is None:
            print("ERROR: Distances are None, cannot compute isolines.")
            return None
        # find the isoline vertices
        isoline_vertices = []
        normal_vectors = []
        # iterate through each face of the mesh
        for face in self.mesh.faces:
            for i in range(3):
                v1_index = face[i]
                v2_index = face[(i + 1) % len(face)]
                v1_distance = distances[v1_index]
                v2_distance = distances[v2_index]

                # check if the target distance is within the face
                if (v1_distance <= target_distance <= v2_distance) or (v2_distance <= target_distance <= v1_distance):
                    # interpolate the vertex position at the target distance
                    t = (target_distance - v1_distance) / (v2_distance - v1_distance)
                    v1_pos = self.mesh.vertices[v1_index]
                    v2_pos = self.mesh.vertices[v2_index]
                    interpolated_vertex = v1_pos + t * (v2_pos - v1_pos)
                    isoline_vertices.append(interpolated_vertex)

                    # compute the normal vector of the face
                    face_normal = self.mesh.face_normals[self.mesh.faces.tolist().index(face.tolist())]
                    normal_vectors.append(face_normal)

        return isoline_vertices, normal_vectors
    

    def compute_isolines(self, target_distance : float, distances : np.ndarray) -> (list, list):
        ''' 
        Compute isolines for the entire surface at specified intervals.
        Parameters:
            target_distance: The distance at which to compute the isolines.
            distances: A numpy array of geodesic distances from the source vertex to all other vertices
        Returns:
            isolines: A list of lists, where each inner list contains the vertices of an isoline at the specified distance.
            isoline_normals: A list of lists, where each inner list contains the normal vectors of the faces corresponding to the isoline vertices.
        '''
        isolines = []
        isoline_normals = []
        max_distance = np.max(distances)
        max_lines = (int)(max_distance / target_distance)

        
        isoline_distance = 0.0
        for i in range(0, max_lines):
            isoline_distance += target_distance
            print(f"Computing isoline for distance: {isoline_distance}")
            isoline, norms = self.find_single_isoline(isoline_distance, distances)
            isolines.append(isoline)
            isoline_normals.append(norms)
    
        return isolines, isoline_normals

    def find_geodesic_paths(self, target_distance: float, source_vertex_indices=None) -> (list, list):
        '''
        Compute geodesic paths (isolines) at specified intervals from the source vertex.
        Parameters:
            target_distance: The distance at which to compute the isolines.
            source_vertex_indices: Optional index of the source vertex in the mesh. If None, uses the closest vertex to self.source_vertex.
        Returns:
            isolines: A list of lists, where each inner list contains the vertices of an isoline at the specified distance.
            isoline_normals: A list of lists, where each inner list contains the normal vectors of the faces corresponding to the isoline vertices.
        '''
        distances = self.compute_geodesic_distances(source_vertex_indices)
        if distances is None:
            print("ERROR: Distances are None, cannot compute geodesic paths.")
            return None, None
        isolines, isoline_normals = self.compute_isolines(target_distance, distances)
        return isolines, isoline_normals


if __name__ == "__main__":
    planner = GeodesicPlanner('/Users/samantha/Desktop/Coding/Thesis/GeodesicSurfaceInformedPlanner/test_geo/01_Hemisphere_base.ply')
                              
    print("=== Testing with centroid-based source vertex ===")
    distances = planner.compute_geodesic_distances()
    if distances is not None:
        print(f"\nFirst 10 distances from vertex 0:")
        for i in range(0, min(10, len(distances))):
            print(f"Vertex {i}: Distance from source = {distances[i]}")

        unique_distances = np.unique(distances)
        print(f"\nTotal unique distance values from vertex 0: {len(unique_distances)}")
        if len(unique_distances) <= 10:
            print("Unique distances:", unique_distances[:10])

    isolines, normals = planner.compute_isolines(0.2, distances)

    for i in range(len(normals)):
        print(f"Isoline {i} has {len(normals[i])} normal vectors.")

    # plot the mesh and color vertices by distance
    if distances is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(planner.mesh.vertices[:,0], planner.mesh.vertices[:,1], planner.mesh.vertices[:,2],
                       c=distances, cmap='viridis')
        fig.colorbar(p)
        
        # plot the vertices of the isoline in red
        if isolines is not None:
            for isoline in isolines:
                isoline = np.array(isoline)
                if isoline.size > 0:
                    ax.scatter(isoline[:,0], isoline[:,1], isoline[:,2], c='r', s=40, label='Isoline')
            
        ax.set_title('Geodesic Distances from Source Vertex')
        plt.show()