#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from geodesic_planner_msgs.srv import ComputeGeodesics
from geometry_msgs.msg import Point


class PlannerClient(Node):
    def __init__(self):
        super().__init__('planner_client')
        self.client = self.create_client(ComputeGeodesics, 'compute_geodesics')
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        
        self.get_logger().info('Connected to planner service.')

    def send_request(self, mesh_file_path, sources=None, spacing=0.1):
        """
        Send a request to compute geodesics.
        
        Args:
            mesh_file_path (str): Path to the mesh file
            sources (list): List of Point objects representing source points
            spacing (float): Spacing parameter for geodesic computation
        
        Returns:
            Response from the service containing geodesic paths
        """
        request = ComputeGeodesics.Request()
        request.mesh_file_path = mesh_file_path
        request.spacing = spacing
        
        # Set sources if provided
        if sources is None:
            sources = []
        request.sources = sources
        
        self.get_logger().info(f'Requesting geodesics for mesh: {mesh_file_path}')
        self.get_logger().info(f'Spacing: {spacing}')
        self.get_logger().info(f'Number of sources: {len(sources)}')
        
        # Call service asynchronously
        self.future = self.client.call_async(request)
        return self.future

    def process_response(self, response):
        """Process and display the response from the service."""
        self.get_logger().info(f'Received {len(response.geodesic_paths)} geodesic paths')
        
        for idx, path in enumerate(response.geodesic_paths):
            self.get_logger().info(f'Path {idx}: {len(path.poses)} poses')
        
        return response


def main(args=None):
    rclpy.init(args=args)
    
    client = PlannerClient()
    
    mesh_file_path = "/home/smith.15485/AA_DEVEL/spray_coating/ws_paint/src/GeodesicSurfaceInformedPlanner/geodesic_planner_ros2/test_geo/mas-fender02-mm_scale_noholes.ply"
    spacing = 0.1
    
    sources = []
    source_point = Point()
    source_point.x = 0.6
    source_point.y = 0.1
    source_point.z = 0.7
    sources.append(source_point)

    # Send request
    future = client.send_request(mesh_file_path, sources, spacing)
    
    # Wait for response
    rclpy.spin_until_future_complete(client, future)
    
    if future.result() is not None:
        response = future.result()
        client.process_response(response)
        client.get_logger().info('Service call successful!')
    else:
        client.get_logger().error(f'Service call failed: {future.exception()}')
    
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
