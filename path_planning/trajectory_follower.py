import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node

from .utils import LineTrajectory

import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from scipy.spatial.transform import Rotation

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 3.0  # FILL IN #
        self.speed = 1.0  # FILL IN #
        self.wheelbase_length = 0.3  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.odom_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.debug_pub = self.create_publisher(PointStamped,
                                               "/debug/target",
                                               1)

        self.initialized_traj = False

    def pose_callback(self, odometry_msg):
        if not self.initialized_traj: 
            return
        
        robot_pose = odometry_msg.pose.pose 
        robot_point = np.array([robot_pose.position.x, robot_pose.position.y])
        trajectory_points = np.array(self.trajectory.points) 

        ## distances to line segments, vectorized 
        ## https://stackoverflow.com/a/58781995
        starts = trajectory_points[:-1]
        ends = trajectory_points[1:]
        vectors = ends - starts
        normalized = np.divide(vectors, (np.hypot(vectors[:,0], vectors[:,1]).reshape(-1,1)))
        parallel_start = np.multiply(starts-robot_point, normalized).sum(axis=1)
        parallel_end = np.multiply(robot_point-ends, normalized).sum(axis=1)
        clamped = np.maximum.reduce([parallel_start, parallel_end, np.zeros(len(parallel_start))])
        start_vectors = robot_point - starts
        perpendicular = start_vectors[:,0] * normalized[:,1] - start_vectors[:,1] * normalized[:,0]
        distances = np.hypot(clamped, perpendicular)

        closest_segment_index = np.argmin(distances)
        
        ## search for lookahead point
        ## https://codereview.stackexchange.com/a/86428
        found = False
        target = None
        for i in range(closest_segment_index, len(starts)):
            P1 = starts[i]
            V = vectors[i]
            a = np.dot(V, V)
            b = 2 * np.dot(V, P1 - robot_point)
            c = np.dot(P1, P1) + np.dot(robot_point, robot_point) - 2*np.dot(P1, robot_point) - self.lookahead**2
            disc = b**2 - 4*a*c
            # self.get_logger().info(str(P1))
            # self.get_logger().info(str(V))
            # self.get_logger().info(str(robot_point))
            # self.get_logger().info(str(disc))
            if disc < 0:
                continue
            sqrt_disc = np.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2*a)
            t2 = (-b - sqrt_disc) / (2*a)
            # self.get_logger().info(str(t1) + " " + str(t2))
            if not (0 <= t1 <= 1): ## or 0 <= t2 <= 1
                continue
            found = True
            target = P1 + t1 * V
            # self.get_logger().info(str(np.linalg.norm(ends[i] - robot_point)))
            self.lookahead = np.sqrt(np.linalg.norm(ends[i] - robot_point)) + 0.2 
            break

        if not found:
            command = AckermannDriveStamped()
            command.header = odometry_msg.header
            command.header.stamp = self.get_clock().now().to_msg()
            command.drive.steering_angle = 0.0
            command.drive.speed = 0.0
            self.drive_pub.publish(command)
            raise Exception("can't find target point")
        
        # visualize target
        pose = PointStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "/map"
        pose.point.x, pose.point.y = target[0], target[1]
        self.debug_pub.publish(pose)

        ## get pure pursuit control action
        orientation = robot_pose.orientation
        r = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        euler = r.as_euler('xyz', degrees=False)
        angle = euler[2] - np.arctan2(target[1]-robot_point[1], target[0]-robot_point[0])
        l1 = np.linalg.norm(target - robot_point)
        delta = np.arctan(2*self.wheelbase_length*np.sin(angle) / l1)
        
        # send command
        command = AckermannDriveStamped()
        command.header = odometry_msg.header
        command.header.stamp = self.get_clock().now().to_msg()
        command.drive.steering_angle = -delta
        command.drive.speed = self.speed
        self.drive_pub.publish(command)


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
