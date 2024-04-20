import rclpy
import random
import math
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, Point, Quaternion
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
from nav_msgs.msg import Odometry

from scipy.spatial.transform import Rotation as R
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
        
        self.map_data = None
        self.resolution = None
        self.orientation = None
        self.position = None

        self.start_x = 0.0
        self.start_y = 0.0 
        self.start_theta = 0.0 
        self.working_trajectory = None
        self.find_traj = True

        self.end_x = 0.0
        self.end_y = 0.0
        self.goal_theta = 0.0

        self.x_min = -60.0
        self.x_max = 25.0
        self.y_min = -20.0
        self.y_max = 50.0

        self.max_steer = math.pi
        self.step_size = 1.5
        self.collision_step_size = 0.05
        self.num_iter = 200
        self.path_width = 0.1
        self.nodes = 0

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

        self.pos_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_cb,
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
        timer_period = 1/10 #seconds
        self.t = 0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if self.map_data is None:
            return
        if not self.find_traj:
            #self.get_logger().info("trajectory already found")
            return
        nodes = []
        self.t = self.t + 1
        for it in range(self.num_iter):
            #generate random point
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            theta = random.uniform(-math.pi, math.pi)

            shortest_dist = 10000
            closest_node = None

            #find nearest node
            if len(nodes) > 0:
                for node in nodes: #find the closest node
                    disp = math.sqrt((node.x-x)**2 + (node.y-y)**2)
                    if disp < shortest_dist: #if this node closer
                        shortest_dist = disp
                        closest_node = node

            #steer
            new_node = self.steer(x, y, theta, closest_node)
            if new_node is not None:
                nodes.append(new_node)
            else:
                continue
            #check if node is within distance of end 
            end_disp = math.sqrt((new_node.x-self.end_x)**2 + (new_node.y-self.end_y)**2)
            if end_disp<self.step_size:
                end_node = self.steer(self.end_x, self.end_y, new_node)
                if end_node is not None: #we have found a trajectory
                    nodes.append(end_node)
                    current_node = end_node
                    trajectory = []
                    while current_node is not None:
                        trajectory.append(current_node)
                        current_node = current_node.parent
                    self.working_trajectory = trajectory
                    self.visualize_tree(trajectory)
                    self.find_traj = False
                    self.get_logger().info("path length: " +str(len(trajectory)))
                    self.nodes = 0
                    return   
        self.visualize_tree(nodes)           
    

    def steer(self, goal_x, goal_y, goal_theta, parent):
        if parent is not None:
            start_x = parent.x
            start_y = parent.y
            start_theta = parent.theta
        else:
            start_x = self.start_x
            start_y = self.start_y
            start_theta = self.start_theta
            
        angle_to_target = math.atan2(goal_y - start_y, goal_x - start_x)
        
        orientation_diff = min(self.max_steer, abs(angle_to_target - start_theta))
        if angle_to_target - start_theta < 0:
            orientation_diff = -orientation_diff
        
        new_orientation = start_theta + orientation_diff
        
        # Calculate the new position using the step size and the new orientation
        new_x = start_x + self.step_size * math.cos(new_orientation)
        new_y = start_y + self.step_size * math.sin(new_orientation)
        
        # Normalize the orientation to stay within [-pi, pi]
        new_orientation = (new_orientation + math.pi) % (2 * math.pi) - math.pi
        
        #check if path is collision free
        collision_free = self.check_collision(new_x, new_y, start_x, start_y)
        if collision_free:
            # Create a new node with the new position and orientation
            new_node = pathNode(new_x, new_y, goal_theta, parent, self.nodes)
            self.nodes += 1
            return new_node
        else:
            return None

    def check_collision(self, new_x, new_y, parent_x, parent_y):
        #path moves from parent to new
        #create a list of all the points we need to check
        total_dist = math.sqrt((new_x-parent_x)**2 + (new_y-parent_y)**2)
        num_iter = int(total_dist/self.collision_step_size)
        points = []

        # Calculate direction vector components and a perpendicular vector for the width
        direction_x = new_x - parent_x
        direction_y = new_y - parent_y
        norm = math.sqrt(direction_x ** 2 + direction_y ** 2)

        # Normalize direction vector
        dir_x = direction_x / norm
        dir_y = direction_y / norm
        
        # Compute perpendicular (normal) vector to the direction
        perp_x = -dir_y
        perp_y = dir_x

        for i in range(num_iter+1):
            # Calculate intermediate point along the line
            t = self.collision_step_size * i / total_dist
            base_x = parent_x + t * direction_x
            base_y = parent_y + t * direction_y
        
            # Add points across the width of the path
            for offset in np.linspace(-self.path_width / 2, self.path_width / 2, num=int(self.path_width/self.collision_step_size)):  
                x = base_x + offset * perp_x
                y = base_y + offset * perp_y
                points.append((x, y))

        no_collision = True
        for point in points:
            x = point[0]
            y = point[1]
            uv = self.xy_to_uv(x,y)
            if self.map_data[uv[1], uv[0]] > 0:
                no_collision = False
                break
        
        return no_collision
    
    def xy_to_uv(self, x, y):
        xtrans = x-self.position.x
        ytrans = y-self.position.y
        xq = self.orientation.x
        yq = self.orientation.y
        zq = self.orientation.z
        wq = self.orientation.w
        rot = R.from_quat((xq,yq,zq,wq))
        eul = rot.as_euler('xyz')
        theta = eul[2]

        # Apply inverse rotation using the negative of the yaw
        x_rotated = xtrans * math.cos(-theta) - ytrans * math.sin(-theta)
        y_rotated = xtrans * math.sin(-theta) + ytrans * math.cos(-theta)

        # Scale (x_rotated, y_rotated) by the inverse of the resolution to get grid coordinates
        u = int(x_rotated / self.resolution)
        v = int(y_rotated / self.resolution)
        return (u,v)

    def map_cb(self, msg):
        self.get_logger().info("got map data")
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.resolution = msg.info.resolution
        self.orientation = msg.info.origin.orientation
        self.position = msg.info.origin.position

    
    def odom_cb(self, msg):
        pass
        

    def pose_cb(self, msg):
        self.start_x = msg.pose.pose.position.x
        self.start_y = msg.pose.pose.position.y
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w

        rot = R.from_quat((x,y,z,w))
        eul = rot.as_euler('xyz')
        self.start_theta = eul[2]
        self.find_traj = True


    def goal_cb(self, msg):
        self.end_x = msg.pose.position.x
        self.end_y = msg.pose.position.y
        x = msg.pose.orientation.x
        y = msg.pose.orientation.y
        z = msg.pose.orientation.z
        w = msg.pose.orientation.w

        rot = R.from_quat((x,y,z,w))
        eul = rot.as_euler('xyz')
        self.end_theta = eul[2]
        self.find_traj = True

    def plan_path(self, start_point, end_point, map):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def visualize_tree(self, nodes):
        paths = PoseArray()
        paths.header.frame_id = "map"
        paths.header.stamp = self.get_clock().now().to_msg()
        for node in nodes:
            rot = R.from_euler("xyz", [0, 0, node.theta])
            quat = rot.as_quat()
            xq = quat[0]
            yq = quat[1]
            zq = quat[2]
            wq = quat[3]
            pose = Pose(position = Point(x=node.x, y=node.y,z=0.1), orientation=Quaternion(x=xq, y=yq, z=zq, w=wq))
            paths.poses.append(pose)

        self.traj_pub.publish(paths)


class pathNode:
    def __init__(self, x, y, theta, parent=None, nodenumber = 0):
        self.x = x 
        self.y = y
        self.theta = theta
        self.nodenumber = nodenumber
        self.parent = parent  # Reference to the parent node

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
