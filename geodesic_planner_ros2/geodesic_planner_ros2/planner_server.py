#!/usr/bin/env python3

import geometry_msgs
import rclpy
from rclpy.node import Node
from geodesic_planner_msgs.srv import ComputeGeodesics
from geodesic_planner_ros2.geodesic_planner import GeodesicPlanner
import numpy as np
import geodesic_planner_ros2.utils as utils 

class PlannerServer(Node):
    def __init__(self):
        super().__init__('planner_server')
        self.srv = self.create_service(ComputeGeodesics, 'compute_geodesics', self.compute_geodesics_callback)
        self.get_logger().info('Planner server ready.')

    def compute_geodesics_callback(self, request, response):
        planner = GeodesicPlanner(request.mesh_file_path)
        source_point = request.sources[0]
        planner.source_vertex = np.array([source_point.x, source_point.y, source_point.z])
        isolines, normals = planner.find_geodesic_paths(request.spacing)

        print(f"Computed {len(isolines)} isolines.")
        print(f"First isoline has {len(isolines[0])} points.")
        print(f"First normal has {len(normals[0])} vectors.")

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
            response.geodesic_paths.append(path)

        
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