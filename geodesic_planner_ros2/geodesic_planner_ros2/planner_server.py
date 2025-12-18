#!/usr/bin/env python3

import geometry_msgs
import rclpy
from rclpy.node import Node
from geodesic_planner_msgs.srv import ComputeGeodesics
from geodesic_planner_ros2.geodesic_planner import GeodesicPlanner
import numpy as np
import geodesic_planner_ros2.utils as utils
import visualization_msgs
from visualization_msgs.msg import Marker, MarkerArray
from geodesic_planner_ros2.path_ordering import order_paths

from geodesic_planner_ros2.line_fitting import fit_parametric_curve, split_isoline_gaps, interpolate_path, get_normals, extrapolate_endpoints, straight_line_interpolation

class PlannerServer(Node):
    def __init__(self):
        super().__init__('planner_server')
        self.srv = self.create_service(ComputeGeodesics, 'compute_geodesics', self.compute_geodesics_callback)
        self.get_logger().info('Planner server ready.')
        self.sources_publisher = self.create_publisher(visualization_msgs.msg.MarkerArray, 'geodesic_sources', 10)
        self.mesh_publisher = self.create_publisher(visualization_msgs.msg.Marker, 'mesh', 10)
        self.timer = self.create_timer(0.5, self.publish_mesh_marker)

        self.mesh_file_path = ""

        self.max_gap = self.declare_parameter('max_segment_gap', 0.15).get_parameter_value().double_value

        self.path_marker_pub = self.create_publisher(visualization_msgs.msg.MarkerArray, 'individual_paths', 10)
        self.publisher = self.create_publisher(geometry_msgs.msg.PoseArray, 'geodesic_paths', 10)

        self.all_markers = visualization_msgs.msg.MarkerArray()

        self.initial_paths = geometry_msgs.msg.PoseArray()
        self.initial_paths.header.frame_id = "world"

        self.initial_paths_publisher = self.create_publisher(geometry_msgs.msg.PoseArray, 'initial_geodesic_paths', 10)

        self.all_paths = geometry_msgs.msg.PoseArray()
        self.all_paths.header.frame_id = "world"

        self.sources = visualization_msgs.msg.MarkerArray()
        self.sources.markers = []

        self.ordered_paths_publisher = self.create_publisher(visualization_msgs.msg.MarkerArray, 'ordered_geodesic_paths', 10)
        self.ordered_paths_ = visualization_msgs.msg.MarkerArray()
        self.ordered_paths_.markers = []

    def compute_geodesics_callback(self, request, response):
        planner = GeodesicPlanner(request.mesh_file_path)
        self.mesh_file_path = request.mesh_file_path
        self.publish_mesh_marker()

        # source_indices = planner.find_nearest_sources(request.sources)
        self.sources.markers = []
        source_points = []
        if len(request.sources) > 0:
            for source in request.sources:
                source_points.append(np.array([source.x, source.y, source.z]))
        else:
            source_points = planner.find_source_points(planner.mesh)
        source_msg = visualization_msgs.msg.MarkerArray()
        i = 0
        for source in source_points:
            marker = visualization_msgs.msg.Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "source_points"
            marker.id = i
            marker.type = visualization_msgs.msg.Marker.SPHERE
            marker.action = visualization_msgs.msg.Marker.ADD
            marker.pose.position.x = float(source[0])
            marker.pose.position.y = float(source[1])
            marker.pose.position.z = float(source[2])
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.r = 0.7
            marker.color.g = 0.5
            marker.color.b = 1.0
            marker.color.a = 1.0
            source_msg.markers.append(marker)
            i += 1

        self.sources = source_msg
        print(f"Computing geodesic paths from {len(self.sources.markers)} source points.")

        # source_points = []
        # for source in source_indices:
        #     source_points.append(np.array(source))
        


        # get initial geodesic isolines
        isolines, iso_norms = planner.find_geodesic_paths(request.spacing, source_points)

        self.initial_paths.poses = []
        for isoline in isolines:
            path_msg = geometry_msgs.msg.PoseArray()
            path_msg.header.frame_id = "world"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            for point in isoline:
                pose = geometry_msgs.msg.Pose()
                pose.position.x = float(point[0])
                pose.position.y = float(point[1])
                pose.position.z = float(point[2])
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.initial_paths.poses.extend(path_msg.poses)

        # split isolines at large gaps
        isoline_segments = []
        isoline_index = 0
        for isoline in isolines:
            if isoline_index == 0: # skip the first isoline, keep whole source path
                isoline_segments.append(isoline)
                isoline_index += 1
                continue
            segments = split_isoline_gaps(isoline, max_gap=self.max_gap)
            print(f"Split isoline into {len(segments)} segments.")
            for segment in segments:
                seg = []
                for point in segment:
                    
                    seg.append(point)
                isoline_segments.append(seg)
            isoline_index += 1

        # fit parametric curves to each isoline segment and interpolate points
        interpolated_paths = []
        interpolated_normals = []

        point_spacing = 0.1
        idx = 0
        for isoline in isoline_segments:
            if len(isoline) < 3:
                interpolated_paths.append(isoline)
                interpolated_normals.append(get_normals(isoline, planner.mesh))
                idx += 1
                continue
            print(f"Fitting line to isoline with {len(isoline)} points.")
            x,y,z = fit_parametric_curve(isoline, smoothing=0.000)

            points_i, norms_i = interpolate_path(x, y, z, point_spacing, planner.mesh)

            print(f"Interpolated {len(points_i)} points with normals.")
            points_i, norms_i = extrapolate_endpoints(points_i, norms_i, extension_length=0.1, point_spacing=point_spacing)
            
            
            interpolated_paths.append(points_i)
            interpolated_normals.append(norms_i)
            idx += 1

        for norms in interpolated_normals:
            smoothed = self.normals_smoothing(norms)
            norms[:] = smoothed

        # liftoff_segments = []
        # ## Apply liftoff to isoline segments
        # for isoline, norm in zip(interpolated_paths, interpolated_normals):
        #     seg = []
        #     for i, point in enumerate(isoline):
        #         p = geometry_msgs.msg.Pose()
        #         p.position.x = float(point[0])
        #         p.position.y = float(point[1])
        #         p.position.z = float(point[2])
        #         quat = utils.z_align_normal(norm[i][0], norm[i][1], norm[i][2])
        #         p.orientation.x = quat[0]
        #         p.orientation.y = quat[1]
        #         p.orientation.z = quat[2]
        #         p.orientation.w = quat[3]
        #         pose = utils.calculate_lift(p, request.liftoff)
        #         seg.append(np.array([pose.position.x, pose.position.y, pose.position.z]))
        #     liftoff_segments.append(seg)

        # interpolated_paths = liftoff_segments
        # # re-interpolate using normals after liftoff
        # interpolated_liftoff_paths = []
        # interpolated_liftoff_normals = []
        # idx = 0
        # for isoline, normals in zip(interpolated_paths, interpolated_normals):
        #     points = []
        #     norms = []
        #     for i in range(1, len(isoline)):
        #         start_point = isoline[i-1]
        #         end_point = isoline[i]
        #         start_norm = normals[i-1]
        #         end_norm = normals[i]

        #         points_i, norms_i = straight_line_interpolation(start_point, start_norm, end_point, end_norm, point_spacing)
        #         points.extend(points_i)
        #         norms.extend(norms_i)

        #     print(f"Re-interpolated {len(points_i)} points with normals after liftoff.")
        #     if idx > 0:
        #         points, norms = extrapolate_endpoints(points, norms, extension_length=0.05, point_spacing=point_spacing)
        #     interpolated_liftoff_paths.append(points)
        #     interpolated_liftoff_normals.append(norms)
        #     idx += 1
        
        # interpolated_paths = interpolated_liftoff_paths
        # interpolated_normals = interpolated_liftoff_normals

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

        print(f"Ordering {len(interpolated_paths)} paths.")
        visit_order, ordered_paths, ordered_normals = order_paths(interpolated_paths, planner.mesh, interpolated_normals)
        print(f"New number of ordered paths: {len(ordered_paths)}, normals: {len(ordered_normals)}")
        ordered_markers = visualization_msgs.msg.MarkerArray()
        ordered_markers.markers = []
        for i in range(len(ordered_paths)):
            path = ordered_paths[i]
            for j in range(len(path)-1):
                marker = visualization_msgs.msg.Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "ordered_geodesic_path"
                marker.id = i * 1000 + j
                marker.type = visualization_msgs.msg.Marker.LINE_STRIP
                marker.action = visualization_msgs.msg.Marker.ADD
                marker.scale.x = 0.02  # Line width
                marker.scale.y = 0.02
                marker.scale.z = 0.02
                marker.color.a = 1.0
                if i == 0:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                elif i > 0:
                    marker.color.r = i / len(path)
                    marker.color.g = 1.0 - (i / len(path))
                    marker.color.b = 0.5

                start_point = geometry_msgs.msg.Point()
                start_point.x = float(path[j][0])
                start_point.y = float(path[j][1])
                start_point.z = float(path[j][2])

                end_point = geometry_msgs.msg.Point()
                end_point.x = float(path[j+1][0])
                end_point.y = float(path[j+1][1])
                end_point.z = float(path[j+1][2])

                marker.points.append(start_point)
                marker.points.append(end_point)

                ordered_markers.markers.append(marker)
        self.ordered_paths_ = ordered_markers


        self.ordered_paths_publisher.publish(self.ordered_paths_)

        # convert ordered paths to PoseArray messages
        geodesic_paths = []
        # convert isolines and normals into pose arrays
        print(f"len paths: {len(ordered_paths)}, len normals: {len(ordered_normals)}")
        for path_points, path_normals in zip(ordered_paths, ordered_normals):
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

        response.success = True
        response.message = "Geodesic paths computed successfully."

        return response
    
    def publish_mesh_marker(self):
        """Publish the mesh as a visualization marker."""
        if self.mesh_file_path == "":
            return
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
        self.initial_paths_publisher.publish(self.initial_paths)
        self.sources_publisher.publish(self.sources)
        self.ordered_paths_publisher.publish(self.ordered_paths_)
        
    def normals_smoothing(self, normals):
        """
        Smooth a list of normals using a simple moving average.
        Args:
            normals (list of np.array): List of normal vectors to be smoothed.
        Returns:
            list of np.array: Smoothed normal vectors.
        """
        smoothed_normals = []
        window_size = 2
        half_window = window_size // 2
        num_normals = len(normals)

        for i in range(num_normals):
            start_idx = max(0, i - half_window)
            end_idx = min(num_normals, i + half_window + 1)
            window_normals = normals[start_idx:end_idx]
            avg_normal = np.mean(window_normals, axis=0)
            avg_normal /= np.linalg.norm(avg_normal)  # Normalize
            smoothed_normals.append(avg_normal)

        return smoothed_normals
        
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