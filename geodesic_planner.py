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
        # self.source_vertex = np.array([0.5, 0.5, 0.5])  # Example hardcoded source vertex
        self.source_vertex = np.array([0, 0, 0])  # Example hardcoded source vertex
        self.solver = pp3d.MeshHeatMethodDistanceSolver(self.mesh.vertices, self.mesh.faces)


    def compute_geodesic_distances(self, source_vertex_index=None) -> np.ndarray:
        '''
        Compute geodesic distances from the source vertex to all other vertices in the mesh.
        Parameters:
            source_vertex_index: Optional index of the source vertex in the mesh. If None, uses the closest vertex to self.source_vertex.
        Returns:
            distances: A numpy array of geodesic distances from the source vertex to all other vertices in the mesh.
        '''
        # find the closest vertex in the mesh to the source vertex
        source_index = self.mesh.kdtree.query(self.source_vertex)[1]
        print("Actual source vertex point:", self.source_vertex)
        print("Source vertex index:", source_index)
        print("Closest vertex coordinates:", self.mesh.vertices[source_index])
        print("Distance between source point and closest vertex:", np.linalg.norm(self.mesh.vertices[source_index] - self.source_vertex))

        # Check if source index is valid
        if source_index >= len(self.mesh.vertices) or source_index < 0:
            print(f"ERROR: Invalid source index {source_index}")
            return None
            
        # compute geodesic distances from the source vertex to all other vertices
        try:
            distances = self.solver.compute_distance(source_index)
        except Exception as e:
            print(f"ERROR in geodesic computation: {e}")
            return None
        
        # Debug: Print some statistics about the distances
        print(f"Min distance: {np.min(distances)}")
        print(f"Max distance: {np.max(distances)}")
        print(f"Mean distance: {np.mean(distances)}")
        print(f"Number of unique distances: {len(np.unique(distances))}")
        
        return distances

if __name__ == "__main__":
    planner = GeodesicPlanner('/home/samubuntu/AA_DEVEL/thesis/src/GeodesicSurfaceInformedPlanner/test_geo/01_Hemisphere_base.ply')
                              
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

    # plot the mesh and color vertices by distance
    if distances is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(planner.mesh.vertices[:,0], planner.mesh.vertices[:,1], planner.mesh.vertices[:,2],
                       c=distances, cmap='viridis')
        fig.colorbar(p)
        plt.show()