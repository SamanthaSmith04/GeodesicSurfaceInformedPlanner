#!/usr/bin/env python3

import geometry_msgs
import rclpy
from rclpy.node import Node
from geodesic_planner_msgs.srv import ComputeGeodesics
from geodesic_planner_ros2.geodesic_planner import GeodesicPlanner
import numpy as np
import geodesic_planner_ros2.utils as utils
import visualization_msgs
from visualization_msgs.msg import Marker 

from geodesic_planner_ros2.downsample import downsample

class PlannerServer(Node):
    def __init__(self):
        super().__init__('planner_server')
        self.srv = self.create_service(ComputeGeodesics, 'compute_geodesics', self.compute_geodesics_callback)
        self.get_logger().info('Planner server ready.')
        self.sources_publisher = self.create_publisher(geometry_msgs.msg.PointStamped, 'geodesic_sources', 10)
        self.mesh_publisher = self.create_publisher(visualization_msgs.msg.Marker, 'mesh', 10)
        self.timer = self.create_timer(5.0, self.publish_mesh_marker)

        self.mesh_file_path = self.declare_parameter('mesh_path', '').get_parameter_value().string_value

    def compute_geodesics_callback(self, request, response):
        planner = GeodesicPlanner(request.mesh_file_path)

        source_indices = planner.find_nearest_sources(request.sources)

        for source in source_indices:
            # self.get_logger().info(f'Received source point: [{source[0]}, {source[1]}, {source[2]}]')
            # publish the source point for debug
            source_msg = geometry_msgs.msg.PointStamped()
            source_msg.header.stamp = self.get_clock().now().to_msg()
            source_msg.header.frame_id = "world"
            source_msg.point.x = float(source[0])
            source_msg.point.y = float(source[1])
            source_msg.point.z = float(source[2])
            self.sources_publisher.publish(source_msg)
            # self.get_logger().info(f'Published source point: [{source[0]}, {source[1]}, {source[2]}]')

        # source_point = request.sources[0]
        # planner.source_vertex = np.array([source_point.x, source_point.y, source_point.z])
        isolines, normals = planner.find_geodesic_paths(request.spacing, request.sources)

        print(f"Computed {len(isolines)} isolines.")
        print(f"First isoline has {len(isolines[0])} points.")
        print(f"First normal has {len(normals[0])} vectors.")

        geodesic_paths = []
        # convert isolines and normals into pose arrays
        for isoline, norm in zip(isolines, normals):
            path = geometry_msgs.msg.PoseArray()
            for vertex, normal in zip(isoline, norm):
                pose = geometry_msgs.msg.Pose()
                pose.position.x = float(vertex[0])
                pose.position.y = float(vertex[1])
                pose.position.z = float(vertex[2])
                
                rot = utils.z_align_normal(normal[0], normal[1], normal[2])
                pose.orientation.x = float(rot[0])
                pose.orientation.y = float(rot[1])
                pose.orientation.z = float(rot[2])
                pose.orientation.w = float(rot[3])
                path.poses.append(pose)
            geodesic_paths.append(path)
        
        for path in geodesic_paths:
            self.get_logger().info(f'Downsampling geodesic path with {len(path.poses)} points.')
            new_path = downsample(request.spacing / 10.0, 360.0, path)

            self.get_logger().info(f'downsampled geodesic path with {len(new_path.poses)} points.')
            response.geodesic_paths.append(new_path)

        # publish the pose array for debug
        self.get_logger().info('Publishing computed geodesic paths.')
        publisher = self.create_publisher(geometry_msgs.msg.PoseArray, 'geodesic_paths', 10)
        all_paths = geometry_msgs.msg.PoseArray()
        all_paths.header.frame_id = "world"
        all_paths.header.stamp = self.get_clock().now().to_msg()
        for path in response.geodesic_paths:
            all_paths.poses.extend(path.poses)
        publisher.publish(all_paths)

        response.success = True
        response.message = "Geodesic paths computed successfully."

        return response
    
    def publish_mesh_marker(self):
        """Publish the mesh as a visualization marker."""
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mesh"
        marker.id = 0
        marker.type = visualization_msgs.msg.Marker.MESH_RESOURCE
        marker.action = visualization_msgs.msg.Marker.ADD
        marker.mesh_resource = "file://" + self.mesh_file_path
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 0.5  # Semi-transparent
        marker.color.r = 140.0/255.0
        marker.color.g = 194.0/255.0
        marker.color.b = 191.0/255.0
        self.mesh_publisher.publish(marker)
    
        
def main(args=None):
    rclpy.init(args=args)
    node = PlannerServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()