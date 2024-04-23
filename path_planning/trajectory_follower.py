import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
import math

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0  # FILL IN #
        self.speed = 0  # FILL IN #
        self.wheelbase_length = 0  # FILL IN #
        

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)

    def find_goal(lookahead, x, y, trajectory)

    def pose_callback(self, odometry_msg):
        raise NotImplementedError

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True

    def get_point(x, y, lookahead, trajectory):


def line_circle_intersection(start, end, circle_origin, r):
    # Unpack the points
    x1, y1 = start
    x2, y2 = end
    cx, cy = circle_origin
    
    # Calculate the direction vector of the line segment
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate the vector from one endpoint of the line to the circle's origin
    ex = cx - x1
    ey = cy - y1
    
    # Calculate the dot product of the direction vector and the vector to the circle's origin
    t = (ex * dx + ey * dy) / (dx * dx + dy * dy)
    
    # Calculate the closest point on the line segment to the circle's origin
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Calculate the distance between the closest point and the circle's origin
    distance = math.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)
    
    # If the distance is less than or equal to the radius, there is an intersection
    if distance <= r:
        # Calculate the intersection points
        d = math.sqrt(r ** 2 - distance ** 2)
        intersection1_x = closest_x + d * (y2 - y1) / math.sqrt(dx ** 2 + dy ** 2)
        intersection1_y = closest_y - d * (x2 - x1) / math.sqrt(dx ** 2 + dy ** 2)
        intersection2_x = closest_x - d * (y2 - y1) / math.sqrt(dx ** 2 + dy ** 2)
        intersection2_y = closest_y + d * (x2 - x1) / math.sqrt(dx ** 2 + dy ** 2)
        
        return [(intersection1_x, intersection1_y), (intersection2_x, intersection2_y)]
    else:
        return None

# Example usage:
start_point = (0, 0)
end_point = (3, 4)
circle_origin = (2, 2)
radius = 1

intersections = line_circle_intersection(start_point, end_point, circle_origin, radius)
print("Intersection points:", intersections)
       

def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
