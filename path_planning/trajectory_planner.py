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
        self.end_theta = 0.0

        self.x_min = -45.0
        self.x_max = 15.0
        self.y_min = -15.0
        self.y_max = 45.0

        self.max_steer = math.pi/2
        self.step_size = 0.5
        self.collision_step_size = 0.25
        self.collision_width_ss = 0.1
        self.num_iter = 400
        self.path_width = 0.4
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
            "/planned_trajectory/path",
            10
        )
        #"/trajectory/current",
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        timer_period = 1/1 #seconds
        self.t = 0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if self.map_data is None:
            return
        #obstacle checking
        # x = self.end_x
        # y= self.end_y
        # uv = self.xy_to_uv(x,y)
        # if self.map_data[uv[1], uv[0]] == 0:
        #     self.get_logger().info("valid point")
        # else:
        #     self.get_logger().info("obstacle")
        # return
        if not self.find_traj:
            #self.get_logger().info("trajectory already found")
            return
        nodes = []
        self.t = self.t + 1
        for it in range(self.num_iter):
            no_valid_point =True
            x = None
            y = None
            theta = None
            while no_valid_point:
                #generate random point
                x = random.uniform(self.x_min, self.x_max)
                y = random.uniform(self.y_min, self.y_max)
                theta = random.uniform(-math.pi/2, math.pi/2)
                #check if point is not in an obstacle
                uv = self.xy_to_uv(x,y)
                if self.map_data[uv[1], uv[0]] == 0:
                    no_valid_point = False


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
                end_node = self.steer(self.end_x, self.end_y, self.end_theta, new_node)
                if end_node is not None: #we have found a trajectory
                    nodes.append(end_node)
                    current_node = end_node
                    trajectory = []
                    while current_node is not None:
                        trajectory.append(current_node)
                        current_node = current_node.parent
                    self.working_trajectory = trajectory
                    self.plan_path(trajectory)
                    self.find_traj = False
                    self.get_logger().info("path length: " +str(len(trajectory)))
                    self.nodes = 0
                    return   
       #pose_paths = self.create_pose_array(nodes)   
       #self.traj_pub.publish(pose_paths)        
    

    def steer(self, goal_x, goal_y, goal_theta, parent):
        if parent is not None:
            start_x = parent.x
            start_y = parent.y
            start_theta = parent.theta
        else:
            start_x = self.start_x
            start_y = self.start_y
            start_theta = self.start_theta

        #check if path is collision free
        collision_free = self.check_collision(goal_x, goal_y, start_x, start_y)

        if collision_free:
            # Create a new node with the new position and orientation
            new_node = pathNode(goal_x, goal_y, goal_theta, parent, self.nodes)
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

        # # Calculate direction vector components
        direction_x = new_x - parent_x
        direction_y = new_y - parent_y

        for i in range(num_iter+1):
            # Calculate intermediate point along the line
            t = self.collision_step_size * i / total_dist
            base_x = parent_x + t * direction_x
            base_y = parent_y + t * direction_y
            points.append((base_x, base_y))
        
        no_collision = True
        for point in points:
            x = point[0]
            y = point[1]
            uv = self.xy_to_uv(x,y)
            if self.map_data[uv[1], uv[0]] > 0:
                self.get_logger().info("COLLISION")
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
        self.trajectory.clear()


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
        self.trajectory.clear()
        self.get_logger().info(str(self.end_x)+" "+str(self.end_y))

    def plan_path(self, path):
        #create trajectory 
        trajectory = self.create_pose_array(path)
        self.trajectory.fromPoseArray(trajectory)
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def create_pose_array(self, nodes):
        nodes.append(pathNode(self.start_x, self.start_y, self.start_theta, nodenumber = 0))
        nodes.reverse()
        paths = PoseArray()
        paths.header.frame_id = "map"
        paths.header.stamp = self.get_clock().now().to_msg()
        last_node = None
        theta = 0.0
        for node in nodes:
            if last_node is not None:
                theta  = math.atan2(node.y - last_node.y, node.x - last_node.x)
            else:
                theta = node.theta
            rot = R.from_euler("xyz", [0, 0, theta])
            quat = rot.as_quat()
            xq = quat[0]
            yq = quat[1]
            zq = quat[2]
            wq = quat[3]
            pose = Pose(position = Point(x=node.x, y=node.y,z=0.1), orientation=Quaternion(x=xq, y=yq, z=zq, w=wq))
            paths.poses.append(pose)
            last_node = node
        return paths
        #self.traj_pub.publish(paths)


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
