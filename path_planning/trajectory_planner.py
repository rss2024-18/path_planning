import rclpy
import random
import math
from rclpy.node import Node
import time


assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Float32MultiArray, Float32
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
from nav_msgs.msg import Odometry
from scipy.ndimage import binary_dilation

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
        self.find_traj = True

        self.end_x = 0.0
        self.end_y = 0.0
        self.end_theta = 0.0

        self.x_min = -60.0
        self.x_max = 25.0
        self.y_min = -15.0
        self.y_max = 45.0

        self.max_steer = math.pi/2
        self.step_size = 0.5
        self.collision_step_size = 1.0
        self.collision_width_ss = 0.1
        self.num_iter = 1000
        self.path_width = 0.4
        self.nodes = 0
        self.radius = 3.0

        self.debug_points = PoseArray()
        self.best_traj = None
        self.best_traj_cost = None
        self.start_time = time.time()
        self.tot_iter = 0.0
        self.total_points = 0.0
        self.total_nodes = 0.0

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
        self.rand_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )
        
        self.data_pub = self.create_publisher(
            Float32MultiArray,
            "/data",
            10
        )
        self.all_data_pub = self.create_publisher(
            Float32MultiArray,
            "/all_data",
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
        timer_period = 1/20 #seconds
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
        self.tot_iter += 1
        for it in range(self.num_iter):
            no_valid_point =True
            x = None
            y = None
            theta = None
            while no_valid_point:
                #generate random point
                x = random.uniform(self.x_min, self.x_max)
                y = random.uniform(self.y_min, self.y_max)
                theta = random.uniform(-math.pi, math.pi)
                #check if point is not in an obstacle
                uv = self.xy_to_uv(x,y)
                if self.map_data[uv[1], uv[0]] == 0:
                    no_valid_point = False
                    self.total_points += 1


            closest_node = None
            cost = None
            shortest_dist = 1000.0
            nodes_in_rad = []

            # if len(nodes)>0: 
            #     for node in nodes: #find the closest node
            #         disp = math.sqrt((node.x-x)**2 + (node.y-y)**2)
            #         if disp < shortest_dist: #if this node closer
            #             shortest_dist = disp
            #             closest_node = node


            #find node within radius with the lowest cost 
            if len(nodes) > 0:
                for node in nodes: #find the best parent node
                    disp = math.sqrt((node.x-x)**2 + (node.y-y)**2)
                    if disp < self.radius: #if node within the radius check the cost of connecting to that node
                        nodes_in_rad.append(node)
                        path, node_cost = self.backtrack(node) #calculate node cost (total path length)
                        if cost is None: #if this node costs less connect to it 
                            closest_node = node
                            cost = node_cost
                        elif node_cost < cost:
                            closest_node = node
                            cost = node_cost

            if cost is None and len(nodes)>0: #if no nodes were within the radius
                for node in nodes: #find the closest node
                    disp = math.sqrt((node.x-x)**2 + (node.y-y)**2)
                    if disp < shortest_dist: #if this node closer
                        shortest_dist = disp
                        closest_node = node

            #steer
            new_node = self.steer(x, y, theta, closest_node)

            if new_node is not None:
                nodes.append(new_node)
                self.total_nodes += 1
            else:
                continue
            
            #if new node is created 
            #check if rewiring any of the nodes in the radius to be from the new node would reduce their cost
            if len(nodes_in_rad) > 0:
                for node_rad in nodes_in_rad:
                    old_path, old_path_cost = self.backtrack(node_rad)
                    new_node_to_node_rad = math.sqrt((new_node.x-node_rad.x)**2 + (new_node.y-node_rad.y)**2)
                    new_path, new_path_cost = self.backtrack(new_node) 
                    new_path_cost += new_node_to_node_rad
                    if new_path_cost < old_path_cost: #if new path costs less check if there would be a collision if using it 
                        no_collision = self.check_collision(node_rad.x, node_rad.y, new_node.x, new_node.y)
                        if no_collision:
                            node_rad.parent = new_node

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

                    #self.debug_points.header.frame_id = "map"
                    #self.debug_points.header.stamp = self.get_clock().now().to_msg()
                    #self.rand_pub.publish(self.debug_points) 
                    #evaluate this trajectory:
                    new_traj, new_traj_cost = self.backtrack(end_node)
                    self.dataPub(new_traj, new_traj_cost)
                    if self.best_traj_cost is None: #this new trajectory is the first we've found
                        self.best_traj_cost = new_traj_cost
                        self.best_traj = trajectory
                        self.runTrajectory(trajectory)
                    elif new_traj_cost < self.best_traj_cost: #or it is better than the one we've been following
                        self.best_traj_cost = new_traj_cost
                        self.best_traj = trajectory
                        self.runTrajectory(trajectory)
                    return   
        # pose_paths = self.create_pose_array(nodes)   
        # self.rand_pub.publish(pose_paths)        
    

    def steer(self, goal_x, goal_y, goal_theta, parent):
        if parent is not None:
            start_x = parent.x
            start_y = parent.y
            start_theta = parent.theta
        else:
            start_x = self.start_x
            start_y = self.start_y
            start_theta = self.start_theta
            parent = pathNode(self.start_x, self.start_y, self.start_theta)

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

        #Calculate direction vector components
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
            if self.map_data[uv[1], uv[0]] != 0:
                no_collision = False
                break
        
        if no_collision:
            for point in points:
                rot = R.from_euler("xyz", [0, 0, 0])
                quat = rot.as_quat()
                xq = quat[0]
                yq = quat[1]
                zq = quat[2]
                wq = quat[3]
                pose = Pose(position = Point(x=point[0], y=point[1],z=0.1), orientation=Quaternion(x=xq, y=yq, z=zq, w=wq))
                #self.debug_points.poses.append(pose)

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
        map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.resolution = msg.info.resolution
        self.orientation = msg.info.origin.orientation
        self.position = msg.info.origin.position

        # Define the structuring element for dilation (a 3x3 square)
        struct_element = np.ones((5,5))
        dilated_map = binary_dilation(map_data, structure=struct_element).astype(np.uint8)
        self.map_data= dilated_map

    
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
        self.reset()
        self.get_logger().info("start " + str(self.start_x)+" "+str(self.start_y))
        #self.debug_points.poses = []


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
        self.reset()
        self.get_logger().info("goal " + str(self.end_x)+" "+str(self.end_y))
        #self.debug_points.poses = []

    def plan_path(self, path):
        #create trajectory 
        self.trajectory.clear()
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

    def backtrack(self, node):
        #get the list of nodes in a path back to the start
        path = []
        path.append(node)
        length = 0.0
        current_node = node
        while current_node.parent is not None:
            path.append(current_node.parent)
            length += math.sqrt((current_node.x-current_node.parent.x)**2 + (current_node.y-current_node.parent.y)**2)
            current_node = current_node.parent

        return (path, length)

    def reset(self):
        #generate a completely new trajectory
        self.find_traj = True
        self.trajectory.clear()
        self.best_traj_cost = None
        self.best_traj = None
    
    def runTrajectory(self, trajectory):
        self.plan_path(trajectory)
        #self.find_traj = False
        self.get_logger().info("path length: " +str(len(trajectory)))
        self.dataPub(trajectory, self.best_traj_cost, best=True)
        self.nodes = 0
    
    def dataPub(self, trajectory, trajectory_cost, best = False):
        msg = Float32MultiArray()
        current_time = time.time()
        elapsed = current_time - self.start_time
        msg.data = [float(elapsed), float(self.tot_iter), float(trajectory_cost), float(len(trajectory)), float(self.total_points), float(self.total_nodes)]
        if best:
            self.data_pub.publish(msg)
            self.all_data_pub.publish(msg)
        else:
            self.all_data_pub.publish(msg)

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
