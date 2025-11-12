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

from geodesic_planner_ros2.line_fitting import fit_parametric_curve, split_isoline_gaps, interpolate_path, get_normals

class PlannerServer(Node):
    def __init__(self):
        super().__init__('planner_server')
        self.srv = self.create_service(ComputeGeodesics, 'compute_geodesics', self.compute_geodesics_callback)
        self.get_logger().info('Planner server ready.')
        self.sources_publisher = self.create_publisher(geometry_msgs.msg.PointStamped, 'geodesic_sources', 10)
        self.mesh_publisher = self.create_publisher(visualization_msgs.msg.Marker, 'mesh', 10)
        self.timer = self.create_timer(0.5, self.publish_mesh_marker)

        self.mesh_file_path = self.declare_parameter('mesh_path', '').get_parameter_value().string_value

        self.path_marker_pub = self.create_publisher(visualization_msgs.msg.MarkerArray, 'individual_paths', 10)
        self.publisher = self.create_publisher(geometry_msgs.msg.PoseArray, 'geodesic_paths', 10)

        self.all_markers = visualization_msgs.msg.MarkerArray()

        self.all_paths = geometry_msgs.msg.PoseArray()
        self.all_paths.header.frame_id = "world"

    def compute_geodesics_callback(self, request, response):
        planner = GeodesicPlanner(request.mesh_file_path)

        source_indices = planner.find_nearest_sources(request.sources)

        for source in source_indices:
            # publish the source point for debug
            source_msg = geometry_msgs.msg.PointStamped()
            source_msg.header.stamp = self.get_clock().now().to_msg()
            source_msg.header.frame_id = "world"
            source_msg.point.x = float(source[0])
            source_msg.point.y = float(source[1])
            source_msg.point.z = float(source[2])
            self.sources_publisher.publish(source_msg)

        # get initial geodesic isolines
        isolines, _ = planner.find_geodesic_paths(request.spacing, request.sources)

        # split isolines at large gaps
        isoline_segments = []
        for isoline in isolines:
            segments = split_isoline_gaps(isoline, max_gap=0.2)
            print(f"Split isoline into {len(segments)} segments.")
            for segment in segments:
                seg = []
                for point in segment:
                    
                    seg.append(point)
                isoline_segments.append(seg)

        # fit parametric curves to each isoline segment and interpolate points
        interpolated_paths = []
        interpolated_normals = []
        for isoline in isoline_segments:
            if len(isoline) < 3:
                interpolated_paths.append(isoline)
                interpolated_normals.append(get_normals(isoline, planner.mesh))
                
                continue
            print(f"Fitting line to isoline with {len(isoline_segments)} points.")
            x,y,z = fit_parametric_curve(isoline, smoothing=0.0)

            points_i, norms_i = interpolate_path(x, y, z, 0.05, planner.mesh)
            interpolated_paths.append(points_i)
            interpolated_normals.append(norms_i)

        # convert interpolated paths to PoseArray messages
        geodesic_paths = []
        # convert isolines and normals into pose arrays
        print(f"len paths: {len(interpolated_paths)}, len normals: {len(interpolated_normals)}")
        for path_points, path_normals in zip(interpolated_paths, interpolated_normals):
            print(f"Path has {len(path_points)} points and {len(path_normals)} normals.")
            path_msg = geometry_msgs.msg.PoseArray()
            path_msg.header.frame_id = "world"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            for i, point in enumerate(path_points):
                pose = geometry_msgs.msg.Pose()
                pose.position.x = float(point[0])
                pose.position.y = float(point[1])
                pose.position.z = float(point[2])
                # get the corresponding normal
                normal = path_normals[i]
                quat = utils.z_align_normal(normal[0], normal[1], normal[2])
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]
                path_msg.poses.append(pose)
            geodesic_paths.append(path_msg)
        
        response.geodesic_paths = geodesic_paths

        # publish the pose array for debug
        self.get_logger().info('Publishing computed geodesic paths.')

        self.all_paths = geometry_msgs.msg.PoseArray()
        self.all_paths.header.frame_id = "world"
        self.all_paths.header.stamp = self.get_clock().now().to_msg()

        self.all_markers = visualization_msgs.msg.MarkerArray()
        self.all_markers.markers = []

        path_index = 0
        for path in response.geodesic_paths:
            self.all_paths.poses.extend(path.poses)
            
            # create a line strip marker for this path (color gradient based on index, base color based on current path index)
            for i in range(len(path.poses)-1):
                marker = visualization_msgs.msg.Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "geodesic_path"
                marker.id = path_index * 1000 + i
                marker.type = visualization_msgs.msg.Marker.LINE_STRIP
                marker.action = visualization_msgs.msg.Marker.ADD
                marker.scale.x = 0.02  # Line width
                marker.scale.y = 0.02
                marker.scale.z = 0.02
                marker.color.a = 1.0
                marker.color.r = float((path_index % 3) == 0) * (1.0 - float(i) / len(path.poses))
                marker.color.g = float((path_index % 3) == 1) * (1.0 - float(i) / len(path.poses))
                marker.color.b = float((path_index % 3) == 2) * (1.0 - float(i) / len(path.poses))

                start_point = path.poses[i].position
                end_point = path.poses[i+1].position

                marker.points.append(start_point)
                marker.points.append(end_point)

                self.all_markers.markers.append(marker)
            path_index += 1
        self.publisher.publish(self.all_paths)
        self.path_marker_pub.publish(self.all_markers)

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

        self.path_marker_pub.publish(self.all_markers)
        self.publisher.publish(self.all_paths)
    
        
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