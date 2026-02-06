#!/usr/bin/env python
import rclpy
from geodesic_planner_msgs.srv import Remesh
from rclpy.node import Node
import os
import pymeshlab

class RemeshingServer(Node):
  def __init__(self):
    super().__init__('remeshing_server')
    self.srv = self.create_service(Remesh, 'remesh_mesh', self.remesh_callback)
    self.get_logger().info("Remeshing server ready.")

  def remesh_callback(self, request, response):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(request.mesh_file_path)
    if request.convert_to_meters:
        self.get_logger().info("Converting mesh units from mm to meters.")
        ms.compute_matrix_from_scaling_or_normalization(axisx=0.001, axisy=0.001, axisz=0.001, alllayers=True)
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(request.percentage_target_edge_length))
    mesh_name = request.mesh_file_path.split('/')[-1]
    save_dir = os.path.expanduser(request.save_path)
    os.makedirs(save_dir, exist_ok=True)

    mesh_name = os.path.basename(request.mesh_file_path)
    new_file_path = os.path.join(save_dir, "remeshed_" + mesh_name)
    if new_file_path.split('.')[-1] != 'ply':
      new_file_path = new_file_path.rsplit('.', 1)[0]
      new_file_path += '.ply'
    ms.save_current_mesh(file_name=new_file_path, save_vertex_normal=True, save_face_color=True)
    response.new_file_path = new_file_path
    self.get_logger().info(f"Remeshed mesh saved to {new_file_path}")
    response.message = "Remeshing completed successfully."
    response.success = True
    return response


def main(args=None):
  rclpy.init()
  remeshing_server = RemeshingServer()
  rclpy.spin(remeshing_server)
          
if __name__ == '__main__':
  main()
  