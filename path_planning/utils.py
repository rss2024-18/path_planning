import rclpy

import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseArray, Point
from std_msgs.msg import Header
import os
from typing import List, Tuple
import json
from tf_transformations import euler_from_quaternion
from collections import deque
import math
from numpy.linalg import inv

EPSILON = 0.00000000001

''' These data structures can be used in the search function
'''


class LineTrajectory:
    """ A class to wrap and work with piecewise linear trajectories. """

    def __init__(self, node, viz_namespace=None):
        self.points: List[Tuple[float, float]] = []
        self.distances = []
        self.has_acceleration = False
        self.visualize = False
        self.viz_namespace = viz_namespace
        self.node = node

        if viz_namespace:
            self.visualize = True
            self.start_pub = self.node.create_publisher(Marker, viz_namespace + "/start_point", 1)
            self.traj_pub = self.node.create_publisher(Marker, viz_namespace + "/path", 1)
            self.end_pub = self.node.create_publisher(Marker, viz_namespace + "/end_pose", 1)

    # compute the distances along the path for all path segments beyond those already computed
    def update_distances(self):
        num_distances = len(self.distances)
        num_points = len(self.points)

        for i in range(num_distances, num_points):
            if i == 0:
                self.distances.append(0)
            else:
                p0 = self.points[i - 1]
                p1 = self.points[i]
                delta = np.array([p0[0] - p1[0], p0[1] - p1[1]])
                self.distances.append(self.distances[i - 1] + np.linalg.norm(delta))

    def distance_to_end(self, t):
        if not len(self.points) == len(self.distances):
            print(
                "WARNING: Different number of distances and points, this should never happen! Expect incorrect results. See LineTrajectory class.")
        dat = self.distance_along_trajectory(t)
        if dat == None:
            return None
        else:
            return self.distances[-1] - dat

    def distance_along_trajectory(self, t):
        # compute distance along path
        # ensure path boundaries are respected
        if t < 0 or t > len(self.points) - 1.0:
            return None
        i = int(t)  # which segment
        t = t % 1.0  # how far along segment
        if t < EPSILON:
            return self.distances[i]
        else:
            return (1.0 - t) * self.distances[i] + t * self.distances[i + 1]

    def addPoint(self, point: Tuple[float, float]) -> None:
        print("adding point to trajectory:", point)
        self.points.append(point)
        self.update_distances()
        self.mark_dirty()

    def clear(self):
        self.points = []
        self.distances = []
        self.mark_dirty()

    def empty(self):
        return len(self.points) == 0

    def save(self, path):
        print("Saving trajectory to:", path)
        data = {}
        data["points"] = []
        for p in self.points:
            data["points"].append({"x": p[0], "y": p[1]})
        with open(path, 'w') as outfile:
            json.dump(data, outfile)

    def mark_dirty(self):
        self.has_acceleration = False

    def dirty(self):
        return not self.has_acceleration

    def load(self, path):
        print("Loading trajectory:", path)

        # resolve all env variables in path
        path = os.path.expandvars(path)

        with open(path) as json_file:
            json_data = json.load(json_file)
            for p in json_data["points"]:
                self.points.append((p["x"], p["y"]))
        self.update_distances()
        print("Loaded:", len(self.points), "points")
        self.mark_dirty()

    # build a trajectory class instance from a trajectory message
    def fromPoseArray(self, trajMsg):
        for p in trajMsg.poses:
            self.points.append((p.position.x, p.position.y))
        self.update_distances()
        self.mark_dirty()
        print("Loaded new trajectory with:", len(self.points), "points")

    def toPoseArray(self):
        traj = PoseArray()
        traj.header = self.make_header("/map")
        for i in range(len(self.points)):
            p = self.points[i]
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            traj.poses.append(pose)
        return traj

    def publish_start_point(self, duration=0.0, scale=0.1):
        should_publish = len(self.points) > 0
        #self.node.get_logger().info("Before Publishing start point")
        if self.visualize and self.start_pub.get_subscription_count() > 0:
            #self.node.get_logger().info("Publishing start point")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 0
            marker.type = 2  # sphere
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = 0
                marker.pose.position.x = self.points[0][0]
                marker.pose.position.y = self.points[0][1]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                # delete marker
                marker.action = 2

            self.start_pub.publish(marker)
        elif self.start_pub.get_subscription_count() == 0:
            self.node.get_logger().info("Not publishing start point, no subscribers")

    def publish_end_point(self, duration=0.0):
        should_publish = len(self.points) > 1
        if self.visualize and self.end_pub.get_subscription_count() > 0:
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 1
            marker.type = 2  # sphere
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = 0
                marker.pose.position.x = self.points[-1][0]
                marker.pose.position.y = self.points[-1][1]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                # delete marker
                marker.action = 2

            self.end_pub.publish(marker)
        elif self.end_pub.get_subscription_count() == 0:
            print("Not publishing end point, no subscribers")

    def publish_trajectory(self, duration=0.0):
        should_publish = len(self.points) > 1
        if self.visualize and self.traj_pub.get_subscription_count() > 0:
            #self.node.get_logger().info("Publishing trajectory")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 2
            marker.type = marker.LINE_STRIP  # line strip
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = marker.ADD
                marker.scale.x = 0.3
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                for p in self.points:
                    pt = Point()
                    pt.x = p[0]
                    pt.y = p[1]
                    pt.z = 0.0
                    marker.points.append(pt)
            else:
                # delete
                marker.action = marker.DELETE
            self.traj_pub.publish(marker)
            print('publishing traj')
        elif self.traj_pub.get_subscription_count() == 0:
            print("Not publishing trajectory, no subscribers")

    def publish_viz(self, duration=0):
        if not self.visualize:
            print("Cannot visualize path, not initialized with visualization enabled")
            return
        self.publish_start_point(duration=duration)
        self.publish_trajectory(duration=duration)
        self.publish_end_point(duration=duration)

    def make_header(self, frame_id, stamp=None):
        if stamp == None:
            stamp = self.node.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header

# class Coordinate2D:
#     """
#     2D coordinate object used in Map representation
#     """
#     def __init__(self, x: float=0.0, y: float=0.0):
#         self.x = x
#         self.y = y

#     def __str__(self):
#         return f"({self.x}, {self.y})"

#     def __eq__(self, other):
#         return self.x == other.x and self.y == other.y
    
#     def scalar_multiply(self, scalar):
#         return Coordinate2D(self.x * scalar, self.y * scalar)

class Map():
    """
    2D map discretization Abstract Data Type
    """
    def __init__(self, msg, node) -> None:
        self.node = node
        self.height = msg.info.height #1300
        self.width = msg.info.width #1730
        self.resolution = msg.info.resolution
        self.orientation = msg.info.origin.orientation
        poseOrientation = [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        self.angles = euler_from_quaternion(poseOrientation)
        self.posePoint = np.array([[msg.info.origin.position.x], [msg.info.origin.position.y], [msg.info.origin.position.z]])
        self.map_translation = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.data = np.reshape(np.array(msg.data), (self.height, self.width))
        self.raw_data = np.array(msg.data).shape
        self.transformation_matrix = None

    def z_axis_rotation_matrix(self, yaw):
        return np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    
    def transformation_matrix_p2r(self):
        rotation_matrix = self.z_axis_rotation_matrix(self.angles[2])
        rotation_matrix[0][2] = self.map_translation[0]
        rotation_matrix[1][2] = self.map_translation[1]
        diagonal_values = [self.resolution, self.resolution, 1]
        scaling_matrix = np.diag(diagonal_values)
        return np.matmul(rotation_matrix, scaling_matrix)

    def pixel_to_real(self, pixelCoord: Tuple[int, int]) -> Tuple[float, float]:
        self.transformation_matrix = self.transformation_matrix_p2r()
        new_pcoord = (pixelCoord[0], pixelCoord[1], 1.0)
        coords = np.matmul(self.transformation_matrix, new_pcoord)
        # self.node.get_logger().info("real_value: " + str(self.discretization(coords[0], coords[1])))
        return self.discretization(coords[0], coords[1])

    def real_to_pixel(self, realCoord: Tuple[float, float]) -> Tuple[int, int]:
        self.transformation_matrix = self.transformation_matrix_p2r()
        new_rcoord = (realCoord[0], realCoord[1], 1.0)
        x = np.matmul(inv(self.transformation_matrix), new_rcoord)
        # self.node.get_logger().info("pixel_value: " + str(x.astype(int)))
        return x.astype(int)
        # x = realCoord[0]
        # y = realCoord[1]
        # xtrans = x-self.position.x
        # ytrans = y-self.position.y
        # xq = self.orientation.x
        # yq = self.orientation.y
        # zq = self.orientation.z
        # wq = self.orientation.w
        # rot = R.from_quat((xq,yq,zq,wq))
        # eul = rot.as_euler('xyz')
        # theta = eul[2]

        # # Apply inverse rotation using the negative of the yaw
        # x_rotated = xtrans * math.cos(-theta) - ytrans * math.sin(-theta)
        # y_rotated = xtrans * math.sin(-theta) + ytrans * math.cos(-theta)

        # # Scale (x_rotated, y_rotated) by the inverse of the resolution to get grid coordinates
        # u = int(x_rotated / self.resolution)
        # v = int(y_rotated / self.resolution)
        # return (u,v)
    
    def get_pixel(self, u, v):
        return self.data[v][u]

    def discretization(self, x, y):
        return (math.floor(x) + 0.5, math.floor(y) + 0.5)

    def get_neighbors(self, x, y):
        neighbors = []
        directions = [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # self.node.get_logger().info("NX, NY: " + str((nx, ny)))
            u, v, b = self.real_to_pixel((nx, ny))
            # self.node.get_logger().info("U, V: " + str((u, v)))
            # self.node.get_logger().info("Free" if self.get_pixel(u, v) == 0 else "Occupied")
            # if True:
            if self.get_pixel(u, v) == 0:
                move_cost = 14 if dx != 0 and dy != 0 else 10  # Diagonal moves cost 14, others cost 10
                neighbors.append((nx, ny, move_cost))
        return neighbors
    
    def heuristic(self, a, b):
        (x1, y1) = a
        (x2, y2) = b
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) * 10
    
    def a_star(self, start, end):
        start = self.discretization(start[0], start[1])
        end = self.discretization(end[0], end[1])

        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
        # Sort and pop the lowest cost item
            frontier.sort(key=lambda x: x[0])
            current = frontier.pop(0)[1]

            if current == end:
                break

            # Correctly unpack all three values returned from get_neighbors
            # self.node.get_logger().info("Neighbors: " + str(self.get_neighbors(current[0], current[1])))
            for nx, ny, move_cost in self.get_neighbors(current[0], current[1]):
                next_pos = (nx, ny)
                new_cost = cost_so_far[current] + move_cost
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(end, next_pos)
                    frontier.append((priority, next_pos))
                    came_from[next_pos] = current

        return self.reconstruct_path(came_from, start, end)

    def reconstruct_path(self, came_from, start, end):
        current = end
        path = []
        while current != start:
            path.append(current)
            current = came_from.get(current)
        path.append(start)
        path.reverse()
        return path