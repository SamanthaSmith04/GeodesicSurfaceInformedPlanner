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

if __name__ == "__main__":
    planner = GeodesicPlanner('test_geo/012349GEO_Geom04_Hemisphere-r01.obj')
    