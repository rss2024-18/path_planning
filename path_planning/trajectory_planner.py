import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory, Map

import time

import numpy as np


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        self.start = None
        self.end = None
        self.map = None

    def map_cb(self, msg):
        self.map = Map(msg, self)
        # self.get_logger().info(self.map)
        
    def pose_cb(self, msg):
        self.start = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        # self.get_logger().info("Start Pose: " + str(self.start))

    def goal_cb(self, msg):
        self.end = (msg.pose.position.x, msg.pose.position.y)
        # self.get_logger().info("End Pose: " + str(self.end))
        if self.map is not None and self.start is not None:
            path = self.plan_path(self.start, self.end, self.map)

    def plan_path(self, start_point, end_point, map):
        path = self.map.a_star(start_point, end_point)
        self.trajectory.clear()
        if path is not None and len(path) > 0:
            for point in path:
                self.trajectory.addPoint(point)
        else:
            # self.get_logger().info("path not found")
            return
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()